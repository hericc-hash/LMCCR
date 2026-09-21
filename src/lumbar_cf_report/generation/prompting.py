from __future__ import annotations
from .core import core_contract
SYSTEM='''你是腰椎MRI临床报告实现器。输入包含两个严格分工的信息通道：
1. LOCKED CLINICAL CORE：唯一允许决定腰椎曲度、5节段椎间盘异常、椎管狭窄、神经根受压的来源，必须严格遵守。
2. CORE-REDACTED LINGUISTIC SCAFFOLD：来自同一患者Direct草稿，但所有Clinical16核心疾病事实已原位遮蔽为<CORE>；它只用于保留患者特异的非核心细节、句序、标点、连接词和报告风格。
要求：不得从<CORE>的位置或SCAFFOLD推测新的Clinical16阳性；不得用SCAFFOLD推翻CORE；应尽量沿用SCAFFOLD中安全的顺序、短语和局部表达，并依据LOCKED CORE自然补齐正确的核心事实。只输出“所见：...\n结论：...”最终报告。'''
INSTRUCTION='以LOCKED CORE决定事实，以CORE-REDACTED SCAFFOLD决定患者特异语言结构；优先保留安全措辞和局部n-gram，生成自然、专业、连贯的最终腰椎MRI报告。'
def messages(core,valid,scaffold):
    user='[LOCKED CLINICAL CORE]\n'+core_contract(core,valid)+'\n\n[CORE-REDACTED LINGUISTIC SCAFFOLD]\n'+(str(scaffold).strip() or '[EMPTY SCAFFOLD]')+'\n\n[INSTRUCTION]\n'+INSTRUCTION
    return [{'role':'system','content':SYSTEM},{'role':'user','content':user}]
