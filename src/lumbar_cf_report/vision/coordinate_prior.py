#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained HR320 residual-attention sagittal keypoint prior."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels:int, reduction:int=8):
        super().__init__(); hidden=max(channels//reduction,4)
        self.net=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(channels,hidden,1),nn.ReLU(inplace=True),nn.Conv2d(hidden,channels,1),nn.Sigmoid())
    def forward(self,x): return x*self.net(x)

class ResBlock(nn.Module):
    def __init__(self,cin:int,cout:int,dilation:int=1,dropout:float=0.0):
        super().__init__(); pad=dilation
        self.conv1=nn.Conv2d(cin,cout,3,padding=pad,dilation=dilation,bias=False); self.bn1=nn.BatchNorm2d(cout)
        self.conv2=nn.Conv2d(cout,cout,3,padding=pad,dilation=dilation,bias=False); self.bn2=nn.BatchNorm2d(cout)
        self.act=nn.ReLU(inplace=True); self.se=SEBlock(cout); self.drop=nn.Dropout2d(dropout) if dropout>0 else nn.Identity()
        self.shortcut=nn.Identity() if cin==cout else nn.Sequential(nn.Conv2d(cin,cout,1,bias=False),nn.BatchNorm2d(cout))
    def forward(self,x):
        y=self.act(self.bn1(self.conv1(x))); y=self.drop(y); y=self.bn2(self.conv2(y)); y=self.se(y)
        return self.act(y+self.shortcut(x))

class AttentionGate(nn.Module):
    def __init__(self,skip_ch:int,gate_ch:int,inter_ch:int):
        super().__init__(); self.skip_proj=nn.Conv2d(skip_ch,inter_ch,1,bias=False); self.gate_proj=nn.Conv2d(gate_ch,inter_ch,1,bias=False)
        self.psi=nn.Sequential(nn.ReLU(inplace=True),nn.Conv2d(inter_ch,1,1),nn.Sigmoid())
    def forward(self,skip,gate):
        if gate.shape[-2:]!=skip.shape[-2:]: gate=F.interpolate(gate,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        return skip*self.psi(self.skip_proj(skip)+self.gate_proj(gate))

class ResAttentionSagittalPriorNet(nn.Module):
    def __init__(self,in_ch:int=1,base_ch:int=32,num_keypoints:int=6,softargmax_beta:float=30.0,dropout:float=0.05):
        super().__init__(); self.softargmax_beta=float(softargmax_beta); c1,c2,c3,c4=base_ch,base_ch*2,base_ch*4,base_ch*8
        self.enc1=nn.Sequential(ResBlock(in_ch,c1,dropout=dropout),ResBlock(c1,c1,dropout=dropout)); self.pool1=nn.MaxPool2d(2)
        self.enc2=nn.Sequential(ResBlock(c1,c2,dropout=dropout),ResBlock(c2,c2,dropout=dropout)); self.pool2=nn.MaxPool2d(2)
        self.enc3=nn.Sequential(ResBlock(c2,c3,dropout=dropout),ResBlock(c3,c3,dropout=dropout)); self.pool3=nn.MaxPool2d(2)
        self.bottleneck=nn.Sequential(ResBlock(c3,c4,1,dropout),ResBlock(c4,c4,2,dropout),ResBlock(c4,c4,4,dropout))
        self.up3=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(c4,c3,1)); self.att3=AttentionGate(c3,c3,max(c3//2,16)); self.dec3=nn.Sequential(ResBlock(c3+c3,c3,dropout=dropout),ResBlock(c3,c3,dropout=dropout))
        self.up2=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(c3,c2,1)); self.att2=AttentionGate(c2,c2,max(c2//2,16)); self.dec2=nn.Sequential(ResBlock(c2+c2,c2,dropout=dropout),ResBlock(c2,c2,dropout=dropout))
        self.up1=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(c2,c1,1)); self.att1=AttentionGate(c1,c1,max(c1//2,16)); self.dec1=nn.Sequential(ResBlock(c1+c1,c1,dropout=dropout),ResBlock(c1,c1,dropout=dropout))
        self.heatmap_head=nn.Sequential(nn.Conv2d(c1,c1,3,padding=1,bias=False),nn.BatchNorm2d(c1),nn.ReLU(inplace=True),nn.Conv2d(c1,num_keypoints,1))
    def softargmax_2d(self,logits):
        b,p,h,w=logits.shape; prob=torch.softmax(logits.view(b,p,-1)*self.softargmax_beta,dim=-1).view(b,p,h,w)
        ys=torch.linspace(0,1,h,device=logits.device,dtype=logits.dtype).view(1,1,h,1); xs=torch.linspace(0,1,w,device=logits.device,dtype=logits.dtype).view(1,1,1,w)
        return torch.stack([(prob*xs).sum((2,3)),(prob*ys).sum((2,3))],dim=-1)
    def forward(self,x):
        x1=self.enc1(x); x2=self.enc2(self.pool1(x1)); x3=self.enc3(self.pool2(x2)); xb=self.bottleneck(self.pool3(x3))
        y3=self.up3(xb); y3=self.dec3(torch.cat([y3,self.att3(x3,y3)],1)); y2=self.up2(y3); y2=self.dec2(torch.cat([y2,self.att2(x2,y2)],1)); y1=self.up1(y2); y1=self.dec1(torch.cat([y1,self.att1(x1,y1)],1))
        logits=self.heatmap_head(y1); return {'heatmap_logits':logits,'coords_norm':self.softargmax_2d(logits)}

def load_prior_checkpoint(model:nn.Module,checkpoint_path:str,device:torch.device)->Dict[str,Any]:
    path=Path(checkpoint_path)
    if not path.is_file(): raise FileNotFoundError(path)
    try: payload=torch.load(str(path),map_location=device,weights_only=False)
    except TypeError: payload=torch.load(str(path),map_location=device)
    state=None
    if isinstance(payload,Mapping):
        for key in ('model','state_dict','model_state','model_state_dict'):
            if isinstance(payload.get(key),Mapping): state=payload[key]; break
        if state is None: state=payload
    else: state=payload
    current=model.state_dict(); loadable={}
    prefixes=('module.','model.','prior_encoder.','prior_net.','network.','net.')
    for key,value in state.items():
        cleaned=str(key); changed=True
        while changed:
            changed=False
            for prefix in prefixes:
                if cleaned.startswith(prefix): cleaned=cleaned[len(prefix):]; changed=True
        if cleaned in current and hasattr(value,'shape') and tuple(value.shape)==tuple(current[cleaned].shape): loadable[cleaned]=value
    missing,unexpected=model.load_state_dict(loadable,strict=False)
    if len(loadable)<int(0.90*len(current)): raise RuntimeError(f'Prior checkpoint coverage too low: loaded={len(loadable)} total={len(current)}')
    args=dict(payload.get('args') or {}) if isinstance(payload,Mapping) else {}
    return {'loaded':len(loadable),'total':len(current),'missing':len(missing),'unexpected':len(unexpected),'checkpoint_args':args}

