"""Three prespecified global blends of independently fitted missing corrections.
Only callback inputs are used; each component receives the original baseline.
"""
from pathlib import Path
import importlib.util
import numpy as np


def _load(name):
    path=Path(__file__).resolve().parent/(name+'.py')
    spec=importlib.util.spec_from_file_location('madinah_missing_ou_'+name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_missing=_load('missing_component')
_ou=_load('ou_component')
OU_CONFIG={'reversion':.02,'qscale':.3,'same_day_weight':3.,'blend':.25,'kinds':['missing']}
WEIGHTS=(1.0, 0.5)  # Selected only from three prespecified global checks.


def predict_fold(train,released,targets,baseline,weights=None):
    weights=WEIGHTS if weights is None else weights
    result=baseline.copy(deep=True)
    selected=result.id.isin(targets.loc[targets.kind.eq('missing'),'id'])
    if not selected.any():
        return result
    # Both corrections are formed relative to this SAME unmodified baseline.
    m=_missing.predict_fold(train,released,targets,baseline.copy(deep=True)).set_index('id').people
    if weights[1]:
        o=_ou.predict_fold(train,released,targets,baseline.copy(deep=True),config=OU_CONFIG).set_index('id').people
    else:
        o=baseline.set_index('id').people
    ids=result.loc[selected,'id']; b=result.loc[selected,'people'].to_numpy(float)
    pred=b+weights[0]*(ids.map(m).to_numpy(float)-b)+weights[1]*(ids.map(o).to_numpy(float)-b)
    result.loc[selected,'people']=np.maximum(0.,pred)
    return result
