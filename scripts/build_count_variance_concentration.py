"""Two-class count distributions from thesis eq:multinomial-mass and eq:dm-mass.

Illustration only. Both classes have mean probability 0.5 and the count total is 20.
The first-class marginal is binomial or beta-binomial, respectively.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binom, betabinom
root=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42})
n=np.arange(21)
R=20
fig,ax=plt.subplots(figsize=(5.3,4.3))
for a0,color in [(4,'#00883a'),(20,'#326a9f'),(100,'#bb741d')]:
    prob=betabinom.pmf(n,R,a0/2,a0/2)
    assert np.isclose(prob.sum(),1)
    assert np.isclose(np.sum(n*prob),10)
    assert np.isclose(np.sum((n-10)**2*prob),R*.25*(a0+R)/(a0+1))
    ax.plot(n,prob,'--',lw=1.7,color=color,label=rf'DM, $\alpha_0={a0}$')
prob=binom.pmf(n,R,.5)
assert np.isclose(np.sum((n-10)**2*prob),5)
ax.plot(n,prob,'-',lw=1.7,color='#000000',label='Multinomial')
ax.set(xlabel=r'Count for class 1, $n_1$',ylabel='Probability',xlim=(0,20),ylim=(0,.19),xticks=[0,5,10,15,20])
ax.set_title(r'Two classes, $q=(0.5,0.5)$, $R=20$',fontsize=11,pad=10)
ax.set_yticks([0,.05,.10,.15])
handles,labels=ax.get_legend_handles_labels()
order=[3,0,1,2]
ax.legend([handles[i] for i in order],[labels[i] for i in order],loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2,columnspacing=1.2,handlelength=2)
fig.subplots_adjust(left=.16,right=.98,top=.9,bottom=.30)
for ext in ('pdf','png'):
    fig.savefig(root/'figures'/f'count_variance_concentration.{ext}',dpi=180,bbox_inches='tight',pad_inches=.05)
plt.close(fig)
print('Created count variance figure. Checked probability sums, means and variances.')
