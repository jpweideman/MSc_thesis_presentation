"""Configured Gamma densities for the thesis standard deviation experiment.
Read-only sources: BNN-EDL/experiments_e1.yaml and base model config.
Density formula: thesis eq:gamma-density. Settings: tab:e2-sd-sweep.
"""
from pathlib import Path
import json
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gamma

root=Path(__file__).resolve().parents[1]
repo=root.parent/'BNN-EDL'
experiments=yaml.safe_load((repo/'experiments_e1.yaml').read_text())['experiments']
base=yaml.safe_load((repo/'configs/cifar10_dirichlet_bnn_sgld.yaml').read_text())['training']['prior_fs']['params']
settings=[]
for percent in [5,20,60]:
    key='e1_c10_L100_shift' if percent==20 else f'e1_c10_L100_sd{percent}_shift'
    params=dict(base)
    for setting in experiments[key]['overrides']:
        for name in ['concentration','rate']:
            prefix=f'training.prior_fs.params.{name}='
            if setting.startswith(prefix): params[name]=float(setting[len(prefix):])
    k,b=float(params['concentration']),float(params['rate'])
    mode=(k-1)/b
    sd=np.sqrt(k)/b
    assert abs(mode-100)<.11
    assert abs(100*sd/mode-percent)<.5
    settings.append({'sd_percent':percent,'shape':k,'rate':b,'mode':mode,'sd':sd})
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(8.4,2.6))
colours=['#326a9f','#00883a','#d9822b']
t=np.linspace(0,250,6000)
for setting,colour in zip(settings,colours):
    k,b=setting['shape'],setting['rate']
    ax.plot(t,gamma.pdf(t,a=k,scale=1/b),color=colour,lw=2,
        label=f"SD {setting['sd_percent']}%")
    assert np.isclose(t[np.argmax(gamma.pdf(t,a=k,scale=1/b))],setting['mode'],atol=.05)
ax.axvline(100,color='#555555',ls=':',lw=1,zorder=0)
ax.set(xlabel=r'Total concentration $\alpha_0$',
    ylabel='Density',xlim=(0,250),ylim=(0,.085))
ax.set_xticks([0,50,100,150,200,250])
fig.subplots_adjust(left=.095,right=.985,top=.85,bottom=.34)
fig.legend(*ax.get_legend_handles_labels(),loc='lower center',
    bbox_to_anchor=(.53,.005),ncol=3,frameon=False,fontsize=11)
for ext in ['pdf','png']:
    fig.savefig(root/'figures'/f'gamma_sd_sweep.{ext}',dpi=180,bbox_inches='tight',pad_inches=.06)
(root/'figures/gamma_sd_sweep.sources.json').write_text(json.dumps({
    'generator':'scripts/build_gamma_sd_sweep.py',
    'source_files':[str(repo/'experiments_e1.yaml'),str(repo/'configs/cifar10_dirichlet_bnn_sgld.yaml')],
    'kind':'Configured Gamma prior densities, not empirical model outputs',
    'settings':settings},indent=2)+'\n')
print('Generated configured Gamma SD comparison. Verified modes and SD percentages against experiment labels.')
