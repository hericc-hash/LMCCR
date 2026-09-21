from __future__ import annotations
import json
from pathlib import Path
from .plan_text import serialize_plan_text,tokenize_control_blocks
from ..runtime.common import clean_report

FOCUSED_PROMPT=("你是一名放射科医师。上方<临床计划>是唯一允许进入语言实现阶段的临床状态。"
"请逐槽执行计划，而不是根据常见共现关系猜测。阳性槽可以写入对应节段；阴性槽不得写成阳性；不同节段不得互相迁移。"
"辅助结构仅在可靠计划节点支持时出现。输出真实、简洁的腰椎MRI聚焦中文报告，必须使用<所见>与<结论>标签。\n报告：")

def make_prompt(tokenizer,device):
    p=tokenizer(FOCUSED_PROMPT,return_tensors='pt',add_special_tokens=False);return {k:v.to(device) for k,v in p.items()}

def generate_from_plan(realizer,tokenizer,prompt,plan,thresholds,max_new_tokens=512,max_control_len=640,device=None):
    control=serialize_plan_text(plan['main_probabilities'][0],plan['main_valid'][0],thresholds,plan['structure_probabilities'][0],True)
    ct=tokenize_control_blocks(tokenizer,[control],max_control_len)
    o=realizer.generate_text(plan['plan_tokens'],ct['input_ids'].to(device),ct['attention_mask'].to(device),prompt['input_ids'],prompt['attention_mask'],max_new_tokens,tokenizer.eos_token_id,tokenizer.pad_token_id,False,.7,.9,1)
    text=clean_report(tokenizer.decode(o['generated_ids'][0].detach().cpu().tolist(),skip_special_tokens=True))
    return text,control

def read_jsonl_map(path,key_fn):
    path=Path(path);out={}
    if not path.is_file():return out
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        r=json.loads(line);k=key_fn(r)
        if k in out:raise RuntimeError(f'duplicate JSONL key {k} in {path}')
        out[k]=r
    return out

