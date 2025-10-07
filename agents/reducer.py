from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Literal

EffectKind = Literal['emit','call_tool','call_llm','checkpoint','halt']

@dataclass
class Effect:
    kind: EffectKind
    payload: Dict[str, Any] = field(default_factory=dict)

@dataclass
class State:
    run_id: str
    step: int = 0
    halted: bool = False
    context: List[Dict[str, Any]] = field(default_factory=list)

def reduce(state: State, event: Dict[str, Any]) -> Tuple[State, List[Effect]]:
    effects: List[Effect] = []
    ekind = event.get('kind')
    if ekind == 'start':
        effects.append(Effect('emit', {'kind':'token','data':'Starting…'}))
        effects.append(Effect('call_llm', {'schema':'plan_next_step'}))
        return state, effects
    if ekind == 'llm_step':
        step = event['step']
        if step.get('done'):
            effects.append(Effect('halt', {'final': step.get('answer', '')}))
        elif 'tool' in step:
            effects.append(Effect('emit', {'kind':'step_planned','step': step}))
            effects.append(Effect('call_tool', step))
        return state, effects
    if ekind == 'tool_result':
        effects.append(Effect('emit', {'kind':'tool_result','result': event['result']}))
        effects.append(Effect('call_llm', {'schema':'plan_next_step'}))
        return state, effects
    if ekind == 'error':
        effects.append(Effect('emit', {'kind':'error','message': event['message']}))
        effects.append(Effect('call_llm', {'schema':'plan_next_step'}))
        return state, effects
    return state, effects
