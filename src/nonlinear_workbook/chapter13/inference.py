"""Small explicit Mamdani inference engine; no implicit controller dynamics."""
from dataclasses import dataclass
from math import prod
import numpy as np
from .sets import FuzzySet,finite,grades


@dataclass(frozen=True)
class Rule:
    antecedents: tuple
    consequent: str
    connective: str='min'
    weight: float=1.

    def __post_init__(self):
        if not self.antecedents or self.connective not in ('min','max','product'): raise ValueError('invalid rule')
        if any(not isinstance(a,(tuple,list)) or len(a)!=2 or any(not isinstance(v,str) or not v for v in a) for a in self.antecedents): raise ValueError('antecedents are named (variable,term) pairs')
        if not isinstance(self.consequent,str) or not self.consequent: raise ValueError('named consequent required')
        object.__setattr__(self,'antecedents',tuple(tuple(a) for a in self.antecedents))
        if np.ndim(self.weight): raise ValueError('scalar rule weight required')
        grades(self.weight)


def infer(inputs,terms,outputs,rules,*,implication='clip',defuzzification='centroid',centroid_mode='continuous',ties='first'):
    if implication not in ('clip','product') or defuzzification not in ('centroid','maximum'): raise ValueError('unknown inference convention')
    if not outputs or not rules: raise ValueError('outputs and rules required')
    if any(not isinstance(rule,Rule) for rule in rules): raise ValueError('Rule instances required')
    reference=next(iter(outputs.values()))
    if not isinstance(reference,FuzzySet): raise ValueError('FuzzySet consequents required')
    if len(reference.universe)*len(rules)>1_000_000: raise ValueError('rule history exceeds one million elements')
    for output in outputs.values(): reference._aligned(output)
    for value in inputs.values():
        if np.ndim(value): raise ValueError('scalar inputs required')
        finite(value)
    histories=[];aggregate=np.zeros_like(reference.membership)
    for rule in rules:
        values=[]
        for variable,term in rule.antecedents:
            if variable not in inputs or variable not in terms or term not in terms[variable]: raise ValueError('missing input or antecedent term')
            fn=terms[variable][term]
            value=fn.at(inputs[variable]) if isinstance(fn,FuzzySet) else fn(inputs[variable])
            if np.ndim(value): raise ValueError('antecedent must return scalar grade')
            values.append(float(grades(value)))
        if rule.consequent not in outputs: raise ValueError('unknown consequent')
        strength=(min(values) if rule.connective=='min' else max(values) if rule.connective=='max' else prod(values))*rule.weight
        shape=outputs[rule.consequent].membership
        activated=np.minimum(shape,strength) if implication=='clip' else shape*strength
        aggregate=np.maximum(aggregate,activated)
        histories.append(dict(antecedent_grades=values,strength=strength,consequent=rule.consequent,membership=activated.tolist()))
    result=FuzzySet(reference.universe,aggregate)
    fired=any(r['strength']>0 for r in histories)
    if not fired: status='no_rule_fired';crisp=None
    elif not np.any(aggregate>0): status='zero_output_mass';crisp=None
    else:
        status='completed'
        crisp=result.centroid(mode=centroid_mode) if defuzzification=='centroid' else result.extremum(ties=ties)
    return dict(status=status,value=crisp,rules=histories,universe=result.universe.tolist(),aggregate=aggregate.tolist(),implication=implication,defuzzification=defuzzification,centroid_mode=centroid_mode,ties=ties)


def singleton_output(values,strengths):
    values=finite(values);s=grades(strengths)
    if values.ndim!=1 or s.shape!=values.shape or not len(values): raise ValueError('matching nonempty vectors required')
    if not np.any(s>0): return dict(status='no_rule_fired',value=None)
    return dict(status='completed',value=float(np.dot(values,s/s.sum())))
