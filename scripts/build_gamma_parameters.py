"""Thesis Gamma prior family with SD approximately 20 percent of the mode.
Sources: Background eq:gamma-density, eq:gamma-mean-variance, eq:gamma-mode;
tables/mode_sweep.tex; BNN-EDL/configs/cifar10_dirichlet_bnn_sgld.yaml.
Theoretical densities for the stated settings, not trained model outputs.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gamma

root=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
shape=27.
modes=[30.,100.,300.]
t=np.linspace(0,550,5000)
fig,ax=plt.subplots(figsize=(8.4,2.6))
colours=['#326a9f','#00883a','#d9822b']
for mode,colour in zip(modes,colours):
    rate=(shape-1)/mode
    density=gamma.pdf(t,a=shape,scale=1/rate)
    assert np.isclose(np.sqrt(shape)/rate/mode,np.sqrt(27)/26)
    assert np.isclose(t[density.argmax()],mode,atol=.12)
    ax.plot(t,density,color=colour,lw=2,
        label=rf'Mode ${mode:g}$, $\beta={rate:.3g}$')
    ax.plot(mode,gamma.pdf(mode,a=shape,scale=1/rate),'o',color=colour,ms=4)
ax.set(xlabel=r'Total concentration $\alpha_0$',
    ylabel='Density',xlim=(0,500),ylim=(0,None))
fig.subplots_adjust(left=.095,right=.985,top=.85,bottom=.34)
fig.legend(*ax.get_legend_handles_labels(),loc='lower center',
    bbox_to_anchor=(.53,.005),ncol=3,frameon=False,fontsize=11)
for ext in ['pdf','png']:
    fig.savefig(root/'figures'/f'gamma_shape_rate.{ext}',dpi=180,bbox_inches='tight',pad_inches=.06)
print('Verified fixed SD/mode ratio and mode locations.')
