#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict,List
import torch
from torch.utils.data import Dataset
from .schema import MAIN_SLOT_NAMES,STRUCTURE_TERMS,FOCUSED_PROMPT


def safe_torch_load(path:str):
    try:return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError:return torch.load(path,map_location="cpu")

class MediationDataset(Dataset):
    def __init__(self,path:str):
        p=Path(path)
        if not p.is_file():raise FileNotFoundError(p)
        self.path=p;self.payload=safe_torch_load(str(p))
        req=["serials","segment_base","task_features","coordinate_state","coordinate_state_raw","task_quality","task_valid","labels","label_valid","global_source","slot_labels","slot_valid","focused_targets","findings","impressions","structure_flags"]
        missing=[k for k in req if k not in self.payload]
        if missing:raise KeyError(f"missing dataset keys: {missing}")
        self.serials=torch.as_tensor(self.payload["serials"]).long();self.segment_base=torch.as_tensor(self.payload["segment_base"]).float();self.task_features=torch.as_tensor(self.payload["task_features"]).float();self.coordinate_state=torch.as_tensor(self.payload["coordinate_state"]).float();self.coordinate_state_raw=torch.as_tensor(self.payload["coordinate_state_raw"]).float();self.task_quality=torch.as_tensor(self.payload["task_quality"]).float();self.task_valid=torch.as_tensor(self.payload["task_valid"]).float();self.labels=torch.as_tensor(self.payload["labels"]).float();self.label_valid=torch.as_tensor(self.payload["label_valid"]).float();self.global_source=torch.as_tensor(self.payload["global_source"]).float();self.slot_labels=torch.as_tensor(self.payload["slot_labels"]).float();self.slot_valid=torch.as_tensor(self.payload["slot_valid"]).float();self.structure_flags=torch.as_tensor(self.payload["structure_flags"]).float()
        self.names=[str(x) for x in self.payload.get("names",[""]*len(self.serials))];self.focused_targets=[str(x) for x in self.payload["focused_targets"]];self.findings=[str(x) for x in self.payload["findings"]];self.impressions=[str(x) for x in self.payload["impressions"]]
        n=len(self.serials)
        if self.segment_base.shape[:2]!=(n,5) or self.task_features.shape[:3]!=(n,5,3) or self.labels.shape!=(n,5,3):raise ValueError("visual tensor contract mismatch")
        if self.slot_labels.shape!=(n,len(MAIN_SLOT_NAMES)) or self.structure_flags.shape!=(n,len(STRUCTURE_TERMS)):raise ValueError("plan label contract mismatch")
    def __len__(self):return int(self.serials.numel())
    def __getitem__(self,i):
        i=int(i);return {"serial":self.serials[i],"name":self.names[i],"segment_base":self.segment_base[i],"task_features":self.task_features[i],"coordinate_state":self.coordinate_state[i],"coordinate_state_raw":self.coordinate_state_raw[i],"task_quality":self.task_quality[i],"task_valid":self.task_valid[i],"labels":self.labels[i],"label_valid":self.label_valid[i],"global_source":self.global_source[i],"slot_labels":self.slot_labels[i],"slot_valid":self.slot_valid[i],"structure_flags":self.structure_flags[i],"focused_target":self.focused_targets[i],"findings":self.findings[i],"impression":self.impressions[i]}
    @property
    def segment_dim(self):return int(self.segment_base.shape[-1])
    @property
    def task_dim(self):return int(self.task_features.shape[-1])
    @property
    def coordinate_dim(self):return int(self.coordinate_state.shape[-1])
    @property
    def global_dim(self):return int(self.global_source.shape[-1])

class FocusedTextCollator:
    def __init__(self,tokenizer,max_text_len:int,prompt:str=FOCUSED_PROMPT):
        self.tokenizer=tokenizer;self.max_text_len=int(max_text_len);self.prompt=str(prompt);self.prompt_ids=tokenizer.encode(self.prompt,add_special_tokens=False)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        if len(self.prompt_ids)>=self.max_text_len-8:raise ValueError("prompt too long")
    def _enc(self,target):
        eos=self.tokenizer.eos_token_id;t=self.tokenizer.encode(str(target),add_special_tokens=False)[:max(1,self.max_text_len-len(self.prompt_ids)-1)];ids=list(self.prompt_ids)+t+([eos] if eos is not None else []);lab=[-100]*len(self.prompt_ids)+t+([eos] if eos is not None else []);return ids,[1]*len(ids),lab
    def __call__(self,batch:List[Dict[str,object]]):
        enc=[self._enc(x["focused_target"]) for x in batch];m=max(len(x[0]) for x in enc);pad=int(self.tokenizer.pad_token_id);ids=[];mask=[];lab=[]
        for a,b,c in enc:
            n=m-len(a);ids.append(a+[pad]*n);mask.append(b+[0]*n);lab.append(c+[-100]*n)
        stack=lambda k:torch.stack([torch.as_tensor(x[k]).float() for x in batch])
        return {"serial":torch.stack([torch.as_tensor(x["serial"]).long() for x in batch]),"names":[x["name"] for x in batch],"segment_base":stack("segment_base"),"task_features":stack("task_features"),"coordinate_state":stack("coordinate_state"),"task_quality":stack("task_quality"),"task_valid":stack("task_valid"),"global_source":stack("global_source"),"slot_labels":stack("slot_labels"),"slot_valid":stack("slot_valid"),"structure_flags":stack("structure_flags"),"input_ids":torch.tensor(ids,dtype=torch.long),"attention_mask":torch.tensor(mask,dtype=torch.long),"labels":torch.tensor(lab,dtype=torch.long),"targets":[x["focused_target"] for x in batch]}

