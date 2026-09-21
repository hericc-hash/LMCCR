from __future__ import annotations
from pathlib import Path
import torch
from .common import torch_load, normalized_serial
from ..clinical.schema import MAIN_SLOT_NAMES, STRUCTURE_TERMS

class MediationCohort:
    """Read-only clinical-mediation cohort used by the report pipeline.

    Direct identifiers such as names are intentionally not exposed. The focused target is kept
    because factual report evaluation requires the frozen reference text.
    """
    def __init__(self, path: str | Path):
        self.path=Path(path)
        if not self.path.is_file(): raise FileNotFoundError(self.path)
        p=torch_load(self.path,'cpu')
        req=['serials','coordinate_state','task_quality','task_valid','labels','label_valid','global_source','slot_labels','slot_valid','focused_targets']
        miss=[k for k in req if k not in p]
        if miss: raise KeyError(f'missing mediation dataset keys: {miss}')
        self.serials=torch.as_tensor(p['serials']).long()
        self.coordinate_state=torch.as_tensor(p['coordinate_state']).float()
        self.task_quality=torch.as_tensor(p['task_quality']).float()
        self.task_valid=torch.as_tensor(p['task_valid']).float()
        self.labels=torch.as_tensor(p['labels']).float()
        self.label_valid=torch.as_tensor(p['label_valid']).float()
        self.global_source=torch.as_tensor(p['global_source']).float()
        self.slot_labels=torch.as_tensor(p['slot_labels']).float()
        self.slot_valid=torch.as_tensor(p['slot_valid']).float()
        self.focused_targets=[str(x) for x in p['focused_targets']]
        n=len(self.serials)
        if self.coordinate_state.shape!=(n,5,47): raise ValueError(f'coordinate_state mismatch {tuple(self.coordinate_state.shape)}')
        if self.task_quality.shape!=(n,5,3) or self.task_valid.shape!=(n,5,3): raise ValueError('quality/valid shape mismatch')
        if self.labels.shape!=(n,5,3) or self.label_valid.shape!=(n,5,3): raise ValueError('label shape mismatch')
        if self.global_source.shape!=(n,256): raise ValueError(f'global_source mismatch {tuple(self.global_source.shape)}')
        if self.slot_labels.shape!=(n,len(MAIN_SLOT_NAMES)) or self.slot_valid.shape!=(n,len(MAIN_SLOT_NAMES)): raise ValueError('main slot shape mismatch')
        self.serial_to_index={normalized_serial(s):i for i,s in enumerate(self.serials)}
        if len(self.serial_to_index)!=n: raise RuntimeError('duplicate serials in mediation dataset')
    def __len__(self): return len(self.serials)
    def item_by_serial(self, serial: int):
        i=self.serial_to_index[int(serial)]
        return self[i]
    def __getitem__(self,i):
        i=int(i)
        return {
            'serial': int(self.serials[i]),
            'coordinate_state': self.coordinate_state[i], 'task_quality': self.task_quality[i], 'task_valid': self.task_valid[i],
            'labels': self.labels[i], 'label_valid': self.label_valid[i], 'global_source': self.global_source[i],
            'slot_labels': self.slot_labels[i], 'slot_valid': self.slot_valid[i], 'focused_target': self.focused_targets[i],
        }

class SelectionCohort:
    """Fields required to reproduce donor availability and report-slot ordering."""
    def __init__(self,path: str | Path):
        self.path=Path(path); p=torch_load(self.path,'cpu')
        for k in ['serials','coordinate_state','task_quality','labels','label_valid']:
            if k not in p: raise KeyError(f'missing {k} in {self.path}')
        self.serials=torch.as_tensor(p['serials']).long(); self.coordinate_state=torch.as_tensor(p['coordinate_state']).float()
        self.task_quality=torch.as_tensor(p['task_quality']).float(); self.labels=torch.as_tensor(p['labels']).float(); self.label_valid=torch.as_tensor(p['label_valid']).float()
    def __len__(self): return len(self.serials)

