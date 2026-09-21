from __future__ import annotations
from typing import List, Sequence
import torch
from .operating_points import LEVELS,TASKS,threshold_for_slot_index

TASK_ZH={"disc":"椎间盘异常","stenosis":"椎管狭窄","nerve":"神经受压"}
STRUCTURE_TERMS=["黄韧带","侧隐窝","椎间孔","硬膜囊","马尾","后纵韧带","神经根管","终丝"]
PRIMARY_STRUCTURE_TERMS=["黄韧带","硬膜囊","马尾"]
SPARSE_STRUCTURE_TERMS=["侧隐窝","椎间孔","终丝"]
UNSUPPORTED_STRUCTURE_TERMS=["后纵韧带","神经根管"]

def disease_slot_index(level_i,task_i): return 1+int(task_i)*5+int(level_i)

def _tolist(x):
    if isinstance(x,torch.Tensor): return x.detach().float().cpu().tolist()
    return list(x)

def serialize_plan_text(main_probabilities,main_valid,thresholds,structure_probabilities=None,include_probabilities=True,primary_structures:Sequence[str]=PRIMARY_STRUCTURE_TERMS):
    p=_tolist(main_probabilities);v=_tolist(main_valid)
    sp=_tolist(structure_probabilities) if structure_probabilities is not None else [0.0]*len(STRUCTURE_TERMS)
    def state_slot(j): return '阳性' if float(p[j])>=threshold_for_slot_index(j,thresholds) else '阴性'
    def state_structure(x): return '阳性' if float(x)>=float(thresholds['structure']) else '阴性'
    lines=['<临床计划>','规则：本计划是唯一允许用于报告生成的临床状态。阳性可写入，阴性不得改写成阳性；不得跨节段补写，不得依据共现关系补写。']
    if v[0]>=.5:
        txt=f"曲度={state_slot(0)}"
        if include_probabilities: txt+=f"(p={p[0]:.3f})"
        lines.append(txt)
    for li,level in enumerate(LEVELS):
        parts=[]
        for ti,task in enumerate(TASKS):
            j=disease_slot_index(li,ti)
            if v[j]<.5: parts.append(f"{TASK_ZH[task]}=无效")
            else:
                s=f"{TASK_ZH[task]}={state_slot(j)}"
                if include_probabilities: s+=f"(p={p[j]:.3f})"
                parts.append(s)
        lines.append(f"{level}："+'；'.join(parts))
    if structure_probabilities is not None:
        items=[]
        for term in primary_structures:
            if term in STRUCTURE_TERMS:
                k=STRUCTURE_TERMS.index(term);s=f"{term}={state_structure(sp[k])}"
                if include_probabilities:s+=f"(p={sp[k]:.3f})"
                items.append(s)
        if items: lines.append('可靠辅助结构：'+'；'.join(items))
        lines.append('稀疏/无监督辅助结构不得主动扩写：'+'、'.join(SPARSE_STRUCTURE_TERMS+UNSUPPORTED_STRUCTURE_TERMS))
    lines += ['执行要求：报告中的疾病阳性/阴性和节段必须与以上主计划逐槽一致。','</临床计划>']
    return '\n'.join(lines)

def render_canonical_report(main_probabilities,main_valid,thresholds,structure_probabilities=None,include_primary_structures=False):
    p=_tolist(main_probabilities);v=_tolist(main_valid);sp=_tolist(structure_probabilities) if structure_probabilities is not None else None
    findings=[];impression=[]
    if v[0]>=.5:
        if p[0]>=float(thresholds['lordosis']): findings.append('腰椎生理曲度变直');impression.append('腰椎生理曲度变直')
        else: findings.append('腰椎生理曲度存在')
    for li,level in enumerate(LEVELS):
        for ti,task in enumerate(TASKS):
            j=disease_slot_index(li,ti)
            if v[j]<.5 or p[j]<threshold_for_slot_index(j,thresholds): continue
            if task=='disc':clause=f'{level}椎间盘异常'
            elif task=='stenosis':clause=f'{level}椎管狭窄'
            else:clause=f'{level}神经根受压'
            findings.append(clause);impression.append(clause)
    if include_primary_structures and sp is not None:
        st=float(thresholds['structure'])
        if sp[STRUCTURE_TERMS.index('黄韧带')]>=st:findings.append('黄韧带增厚')
        if sp[STRUCTURE_TERMS.index('硬膜囊')]>=st:findings.append('硬膜囊受压')
        if sp[STRUCTURE_TERMS.index('马尾')]>=st:findings.append('马尾受压')
    if not findings:findings=['腰椎本次聚焦计划未提示明确目标异常']
    if not impression:impression=['本次聚焦计划未提示明确目标病变']
    return '<所见>'+'。'.join(findings)+'。</所见><结论>'+'；'.join(impression)+'。</结论>'

def tokenize_control_blocks(tokenizer,texts,max_length=640):
    return tokenizer(list(texts),return_tensors='pt',padding=True,truncation=True,max_length=int(max_length),add_special_tokens=False)

