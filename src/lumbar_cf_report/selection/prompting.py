from __future__ import annotations
from .core import core_contract
SYSTEM='''你是腰椎MRI临床报告实现器。输入包含两个严格分工的信息通道：\n1. LOCKED CLINICAL CORE：唯一允许决定腰椎曲度、5节段椎间盘异常、椎管狭窄、神经根受压的来源。\n2. CORE-REDACTED LINGUISTIC SCAFFOLD：只提供患者特异的非核心细节、句序、标点、连接词和报告风格。\n不得从<CORE>推测新的Clinical16阳性，不得用SCAFFOLD推翻CORE。应尽量保留安全措辞和局部n-gram。只输出“所见：...\\n结论：...”最终报告。'''
INSTRUCTION='以LOCKED CORE决定事实，以CORE-REDACTED SCAFFOLD决定患者特异语言结构；优先保留安全措辞和局部n-gram，生成自然、专业、连贯的腰椎MRI报告。'
def messages(core,valid,scaffold):
    user='[LOCKED CLINICAL CORE]\n'+core_contract(core,valid)+'\n\n[CORE-REDACTED LINGUISTIC SCAFFOLD]\n'+(str(scaffold).strip() or '[EMPTY SCAFFOLD]')+'\n\n[INSTRUCTION]\n'+INSTRUCTION
    return [{'role':'system','content':SYSTEM},{'role':'user','content':user}]
