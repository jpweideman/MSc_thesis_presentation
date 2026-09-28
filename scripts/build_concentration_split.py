"""Dirichlet categorical uncertainty terms across total concentration.

Generic closed-form illustration of thesis Propositions 2.1, 2.2 and Corollary
2.3. The limit at zero belongs to the unrestricted Dirichlet family. At the
chosen mean the softplus EDL parameterisation only permits alpha_0 > 2.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import digamma

root=Path(__file__).resolve().parents[1]
source=root.parent/'MSc_thesis/chapters/background.tex'
mean=np.array([.5,.5])
totals=np.geomspace(.001,1000,1200)
predictive=float(-np.sum(mean*np.log(mean)))
aleatoric=np.sum(mean[None,:]*(digamma(totals[:,None]+1)-digamma(totals[:,None]*mean[None,:]+1)),axis=1)
distributional=predictive-aleatoric
assert np.all(np.diff(aleatoric)>0) and np.all(np.diff(distributional)<0)
assert np.all(aleatoric>0) and np.all(distributional>0)
assert np.allclose(aleatoric+distributional,predictive)
assert aleatoric[0]<.001 and distributional[-1]<.001
BLUE,ORANGE='#326a9f','#d9822b'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#aaaaaa',
 'xtick.color':'#333333','ytick.color':'#333333','pdf.fonttype':42,'savefig.facecolor':'white'})
fig,ax=plt.subplots(figsize=(8.4,3.0))
ax.axhline(predictive,color='#333333',ls='--',lw=1.8,label='Predictive (sum)',zorder=3)
ax.plot(totals,aleatoric,color=BLUE,lw=2.8,label='Aleatoric')
ax.plot(totals,distributional,color=ORANGE,lw=2.8,label='Distributional')
ax.set(xscale='log',xlim=(totals[0],totals[-1]),ylim=(0,predictive*1.12),
 xlabel=r'Total concentration $\alpha_0$',ylabel='Uncertainty')
ax.set_xticks([.001,.1,10,1000],[r'$10^{-3}$',r'$10^{-1}$',r'$10^1$',r'$10^3$'])
ax.set_yticks([0,predictive/2,predictive],['0',r'$\frac{1}{2}\log 2$',r'$\log 2$'])
ax.minorticks_off()
fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.54,.01),ncol=3,frameon=False,fontsize=12)
fig.subplots_adjust(left=.105,right=.975,top=.96,bottom=.36)
for ext in ['pdf','png']:
 fig.savefig(root/f'figures/concentration_split_comparison.{ext}',dpi=180,bbox_inches='tight',pad_inches=.06)
plt.close(fig)
record={'generator':'scripts/build_concentration_split.py',
 'kind':'Generic Dirichlet closed-form illustration, not experimental results',
 'mean':mean.tolist(),'total_concentration_grid':[.001,1000,1200,'log spaced'],
 'predictive_entropy_nats':predictive,'aleatoric_endpoints':aleatoric[[0,-1]].tolist(),
 'distributional_endpoints':distributional[[0,-1]].tolist(),
 'edl_strict_lower_bound_at_this_mean':float(1/mean.min()),
 'limits':'As alpha_0 tends to zero, aleatoric tends to zero. As alpha_0 tends to infinity, distributional tends to zero. Their sum stays at predictive entropy.',
 'scope':'General Dirichlet family. EDL restricts the comparison to alpha_0 > 2. No change in a weight posterior is depicted.',
 'thesis_labels':['eq:aleatoric-as-function-of-total-concentration','eq:dirichlet-predictive-uncertainty',
 'eq:distributional-remainder','prop:aleatoric-monotonicity','prop:aleatoric-limits','cor:distributional-monotonicity'],
 'thesis_source':str(source),'thesis_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
(root/'figures/concentration_split_comparison.sources.json').write_text(json.dumps(record,indent=2)+'\n')
print('Generated continuous uncertainty curves. Verified monotonicity, fixed sum and numerical approach to both limits.')

# EDL restriction for the same mean. Boundary values are limits, not attainable outputs.
edl_totals=np.geomspace(2.0+1e-6,1000.,1000)
edl_aleatoric=np.sum(mean[None,:]*(digamma(edl_totals[:,None]+1)-digamma(edl_totals[:,None]*mean[None,:]+1)),axis=1)
edl_distributional=predictive-edl_aleatoric
assert np.all(edl_totals[:,None]*mean[None,:]>1)
assert np.allclose(edl_aleatoric+edl_distributional,predictive)
assert np.all(np.diff(edl_aleatoric)>0) and np.all(np.diff(edl_distributional)<0)
assert np.isclose(edl_aleatoric[0],.5,atol=1e-6)
fig,ax=plt.subplots(figsize=(8.4,3.0))
ax.axhline(predictive,color='#333333',ls='--',lw=1.8,label='Predictive (sum)',zorder=3)
ax.plot(edl_totals,edl_aleatoric,color=BLUE,lw=2.8,label='Aleatoric')
ax.plot(edl_totals,edl_distributional,color=ORANGE,lw=2.8,label='Distributional')
for value,color in [(.5,BLUE),(predictive-.5,ORANGE)]:
 ax.plot(2,value,'o',mfc='white',mec=color,mew=1.8,ms=7,zorder=5,clip_on=False)
ax.set(xscale='log',xlim=(2,1000),ylim=(0,predictive*1.12),
 xlabel=r'Total concentration $\alpha_0$',ylabel='Uncertainty')
ax.set_xticks([2,10,100,1000],['2','10',r'$10^2$',r'$10^3$'])
ax.set_yticks([0,predictive/2,predictive],['0',r'$\frac{1}{2}\log 2$',r'$\log 2$'])
ax.minorticks_off()
fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.54,.01),ncol=3,frameon=False,fontsize=12)
fig.subplots_adjust(left=.105,right=.975,top=.96,bottom=.36)
for ext in ['pdf','png']:
 fig.savefig(root/f'figures/concentration_split_edl.{ext}',dpi=180,bbox_inches='tight',pad_inches=.06)
plt.close(fig)
edl_record=dict(record)
edl_record.update({'kind':'Same Dirichlet split restricted to the softplus EDL parameterisation',
 'total_concentration_grid':[float(edl_totals[0]),float(edl_totals[-1]),len(edl_totals),'log spaced'],
 'aleatoric_endpoints':edl_aleatoric[[0,-1]].tolist(),'distributional_endpoints':edl_distributional[[0,-1]].tolist(),
 'boundary_at_2':'Excluded for softplus. Open circles show limiting aleatoric=0.5 and distributional=log(2)-0.5.',
 'scope':'Same fixed mean and conditional decomposition. Every computed concentration component is greater than one.'})
(root/'figures/concentration_split_edl.sources.json').write_text(json.dumps(edl_record,indent=2)+'\n')
print('Generated EDL-only plot. Verified alpha_c > 1, unchanged sum and boundary limits.')
