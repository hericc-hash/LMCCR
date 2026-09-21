#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disc evidence model with a global/core shared-encoder architecture."""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def _gn(ch:int)->nn.GroupNorm:
    g=min(16,int(ch))
    while g>1 and ch%g: g-=1
    return nn.GroupNorm(g,int(ch))

def _wrap(a:torch.Tensor)->torch.Tensor:
    a=torch.atan2(torch.sin(a),torch.cos(a)); a=torch.where(a>0.5*torch.pi,a-torch.pi,a); return torch.where(a<-0.5*torch.pi,a+torch.pi,a)

class DiscEncoder(nn.Module):
    def __init__(self,in_channels:int=4,hidden_dim:int=256,dropout:float=0.15):
        super().__init__(); from torchvision.models import resnet18
        base=resnet18(weights=None,norm_layer=_gn); old=base.conv1; base.conv1=nn.Conv2d(in_channels,old.out_channels,kernel_size=old.kernel_size,stride=old.stride,padding=old.padding,bias=False); nn.init.kaiming_normal_(base.conv1.weight,mode='fan_out',nonlinearity='relu')
        self.backbone=nn.Sequential(base.conv1,base.bn1,base.relu,base.maxpool,base.layer1,base.layer2,base.layer3,base.layer4); self.pool=nn.AdaptiveAvgPool2d(1); self.proj=nn.Sequential(nn.Flatten(),nn.Linear(512,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU(),nn.Dropout(dropout))
    def forward(self,x): return self.proj(self.pool(self.backbone(x.float())))

class DiscEvidenceExpert(nn.Module):
    VERSION='ManualCenter-Disc-Evidence-v1'
    def __init__(self,hidden_dim:int=256,dropout:float=0.15,roi_crop_size:int=192,crop_encoder_batch_size:int=20,use_roi_mask_channel:bool=True,roi_pad_value:float=0.0,roi_min_half_size:float=0.012,
                 disc_posterior_shift:float=0.08,disc_half_w:float=0.64,disc_half_h:float=0.23,posterior_core_center_shift_frac:float=0.40,posterior_core_width_frac:float=0.60,posterior_core_height_frac:float=1.00,
                 disc_angle_clip_deg:float=35.0,l5s1_angle_gain:float=1.70,l5s1_angle_offset_deg:float=-2.0,l5s1_angle_clip_deg:float=70.0,disc_angle_smooth:bool=True):
        super().__init__(); self.hidden_dim=int(hidden_dim); self.roi_crop_size=int(roi_crop_size); self.crop_encoder_batch_size=int(crop_encoder_batch_size); self.use_roi_mask_channel=bool(use_roi_mask_channel); self.roi_pad_value=float(roi_pad_value); self.roi_min_half_size=float(roi_min_half_size)
        self.disc_posterior_shift=float(disc_posterior_shift); self.disc_half_w=float(disc_half_w); self.disc_half_h=float(disc_half_h); self.posterior_core_center_shift_frac=float(posterior_core_center_shift_frac); self.posterior_core_width_frac=float(posterior_core_width_frac); self.posterior_core_height_frac=float(posterior_core_height_frac)
        self.disc_angle_clip_deg=float(disc_angle_clip_deg); self.l5s1_angle_gain=float(l5s1_angle_gain); self.l5s1_angle_offset_deg=float(l5s1_angle_offset_deg); self.l5s1_angle_clip_deg=float(l5s1_angle_clip_deg); self.disc_angle_smooth=bool(disc_angle_smooth)
        self.encoder=DiscEncoder(4 if self.use_roi_mask_channel else 3,self.hidden_dim,dropout)
        self.head=nn.Sequential(nn.LayerNorm(3*self.hidden_dim),nn.Linear(3*self.hidden_dim,self.hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(self.hidden_dim,self.hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(self.hidden_dim,1))
    def _neighbors(self,c):
        first=c[:,1:2]-c[:,0:1]; prev=torch.cat([c[:,0:1]-first,c[:,:-1]],1); last=c[:,-1:]-c[:,-2:-1]; foll=torch.cat([c[:,1:],c[:,-1:]+last],1); gap=0.5*((c[...,1]-prev[...,1]).abs()+(foll[...,1]-c[...,1]).abs()); return gap.clamp_min(1/384),foll-prev
    def _angles(self,t):
        a=_wrap(torch.atan2(t[...,1],t[...,0])-0.5*torch.pi)
        if self.disc_angle_smooth:
            s=a.clone(); s[:,0]=.75*a[:,0]+.25*a[:,1]; s[:,1:-1]=.25*a[:,:-2]+.5*a[:,1:-1]+.25*a[:,2:]; s[:,-1]=.25*a[:,-2]+.75*a[:,-1]; a=_wrap(s)
        clip=self.disc_angle_clip_deg*torch.pi/180; a=a.clamp(-clip,clip).clone(); l5c=self.l5s1_angle_clip_deg*torch.pi/180; off=self.l5s1_angle_offset_deg*torch.pi/180; a[:,4]=(a[:,4]*self.l5s1_angle_gain+off).clamp(-l5c,l5c); return a
    def _clip(self,centers,half,angles):
        centers=centers.clamp(.01,.99); edge=torch.minimum(centers-.002,.998-centers).clamp_min(self.roi_min_half_size); return centers,torch.minimum(half,edge).clamp_min(self.roi_min_half_size),angles
    def compute_disc_geometry(self,coords):
        gap,tan=self._neighbors(coords); ang=self._angles(tan); axis=torch.stack([torch.cos(ang),torch.sin(ang)],-1)
        gc=coords+self.disc_posterior_shift*gap[...,None]*axis; gh=torch.stack([self.disc_half_w*gap,self.disc_half_h*gap],-1).clamp_min(self.roi_min_half_size)
        pc=gc+self.posterior_core_center_shift_frac*gh[...,0:1]*axis; ph=torch.stack([self.posterior_core_width_frac*gh[...,0],self.posterior_core_height_frac*gh[...,1]],-1).clamp_min(self.roi_min_half_size)
        gc,gh,_=self._clip(gc,gh,ang); pc,ph,_=self._clip(pc,ph,ang); return gc,gh,pc,ph,ang
    def extract_crops(self,images,centers,half,angles,valid):
        b,c,h,w=images.shape; l=centers.shape[1]; s=self.roi_crop_size; dtype=images.dtype; device=images.device; imgs=images[:,None].expand(-1,l,-1,-1,-1)
        line=torch.linspace(-1,1,s,device=device,dtype=dtype); gy,gx=torch.meshgrid(line,line,indexing='ij'); base=torch.stack([gx,gy],-1).view(1,1,s,s,2)
        hx=half[...,0:1].clamp_min(self.roi_min_half_size); hy=half[...,1:2].clamp_min(self.roi_min_half_size); mh=torch.maximum(hx,hy); mask=((base[...,0].abs()<=(hx/mh)[...,None])&(base[...,1].abs()<=(hy/mh)[...,None])).to(dtype)
        lx=base[...,0]*mh[...,None]; ly=base[...,1]*mh[...,None]; co=torch.cos(angles)[...,None,None]; si=torch.sin(angles)[...,None,None]; dx=lx*co-ly*si; dy=lx*si+ly*co; grid=(centers[...,None,None,:]+torch.stack([dx,dy],-1))*2-1
        sampled=F.grid_sample(imgs.reshape(b*l,c,h,w),grid.reshape(b*l,s,s,2),mode='bilinear',padding_mode='zeros',align_corners=True).reshape(b,l,c,s,s); mask=mask[:,:,None]; sampled=sampled*mask; gate=valid.to(dtype)[:,:,None,None,None]; sampled=sampled*gate; mask=mask*gate
        return torch.cat([sampled,mask],2) if self.use_roi_mask_channel else sampled
    def _encode(self,crops):
        b,l,k,c,h,w=crops.shape; flat=crops.reshape(b*l*k,c,h,w); out=[]
        for st in range(0,len(flat),self.crop_encoder_batch_size): out.append(self.encoder(flat[st:st+self.crop_encoder_batch_size]))
        return torch.cat(out,0).reshape(b,l,k,self.hidden_dim)
    def forward(self,sagittal_images,sagittal_coords,sagittal_valid)->Dict[str,torch.Tensor]:
        gc,gh,pc,ph,a=self.compute_disc_geometry(sagittal_coords); global_crop=self.extract_crops(sagittal_images,gc,gh,a,sagittal_valid); posterior_crop=self.extract_crops(sagittal_images,pc,ph,a,sagittal_valid); tok=self._encode(torch.stack([global_crop,posterior_crop],2)); g=tok[:,:,0]; p=tok[:,:,1]; feat=torch.cat([g,p,g-p],-1); b,l=feat.shape[:2]; logits=self.head(feat.reshape(b*l,-1)).reshape(b,l)
        return {'disc_visual_logits':logits,'disc_tokens':tok,'disc_global_crops_preview':global_crop.detach(),'disc_posterior_crops_preview':posterior_crop.detach(),'disc_global_centers':gc,'disc_global_half_sizes':gh,'disc_posterior_centers':pc,'disc_posterior_half_sizes':ph,'disc_angles':a}

