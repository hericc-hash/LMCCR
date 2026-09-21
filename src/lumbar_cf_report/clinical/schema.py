#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
import torch

VERSION = "Plan-Faithful-Counterfactual-Report-Realizer-v1"
DATASET_VERSION = "Explicit-Clinical-Mediation-Dataset-v1"
LEVELS: List[str] = ["L1/2", "L2/3", "L3/4", "L4/5", "L5/S1"]
TASKS: List[str] = ["disc", "stenosis", "nerve"]
TASK_ZH: Dict[str, str] = {"disc":"椎间盘异常", "stenosis":"椎管狭窄", "nerve":"神经受压"}
MAIN_SLOT_NAMES: List[str] = ["lordosis"] + [f"{task}:{level}" for task in TASKS for level in LEVELS]
STRUCTURE_TERMS: List[str] = ["黄韧带", "侧隐窝", "椎间孔", "硬膜囊", "马尾", "后纵韧带", "神经根管", "终丝"]
PLAN_NODE_NAMES: List[str] = MAIN_SLOT_NAMES + [f"structure:{x}" for x in STRUCTURE_TERMS]
TASK_STRUCTURE_RELATIONS: Dict[str, List[str]] = {
    "disc": ["后纵韧带", "硬膜囊"],
    "stenosis": ["黄韧带", "侧隐窝", "椎间孔", "硬膜囊", "神经根管"],
    "nerve": ["神经根管", "椎间孔", "马尾", "终丝"],
}
FOCUSED_PROMPT = (
    "你是一名放射科医师。上方<临床计划>是唯一允许进入语言实现阶段的临床状态。"
    "请逐槽执行计划，而不是根据常见共现关系猜测。"
    "阳性槽可以写入对应节段；阴性槽不得写成阳性；不同节段不得互相迁移。"
    "辅助结构仅在可靠计划节点支持时出现。"
    "输出真实、简洁的腰椎MRI聚焦中文报告，必须使用<所见>与<结论>标签。\n报告："
)


def disease_slot_index(level_i:int, task_i:int) -> int:
    return 1 + int(task_i) * len(LEVELS) + int(level_i)

def disease_slot_to_level_task(slot_i:int) -> Tuple[int,int]:
    s=int(slot_i)
    if s <= 0 or s >= len(MAIN_SLOT_NAMES): raise ValueError(f"not a disease slot: {s}")
    off=s-1; return off % len(LEVELS), off // len(LEVELS)

def build_structure_relation_mask() -> torch.Tensor:
    m=torch.zeros(len(STRUCTURE_TERMS), len(MAIN_SLOT_NAMES), dtype=torch.float32)
    for si,term in enumerate(STRUCTURE_TERMS):
        for ti,task in enumerate(TASKS):
            if term in TASK_STRUCTURE_RELATIONS[task]:
                for li in range(len(LEVELS)): m[si,disease_slot_index(li,ti)] = 1.0
    return m

def related_structure_indices(task_i:int) -> List[int]:
    task=TASKS[int(task_i)];allowed=set(TASK_STRUCTURE_RELATIONS[task]);return [i for i,x in enumerate(STRUCTURE_TERMS) if x in allowed]

