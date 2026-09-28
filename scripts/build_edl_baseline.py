"""Illustrate the zero evidence baseline from thesis eq:dirichlet-density.
Equal-sized reproducible samples from Dirichlet densities, not model results.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
vertices = np.array([[0., 0.], [1., 0.], [.5, np.sqrt(3)/2]])
fig, axes = plt.subplots(1, 3, figsize=(12, 3.65))
settings = [(.08, r'$\mathrm{Dir}(\varepsilon,\varepsilon,\varepsilon)$', 'Near vertices as '+r'$\varepsilon\to0^+$'),
            (1., r'$\mathrm{Dir}(1,1,1)$', 'Uniform'),
            (5., r'$\mathrm{Dir}(5,5,5)$', 'Concentrated near centre')]
for ax, (a, title, caption) in zip(axes, settings):
    samples = np.random.default_rng(417).dirichlet(np.full(3, a), 1000)
    assert np.allclose(samples.sum(axis=1), 1)
    points = samples @ vertices
    ax.scatter(points[:,0], points[:,1], s=5, c='#00883a', alpha=.42, edgecolors='none')
    mean = np.full(3, 1/3) @ vertices
    mean_dot = ax.scatter(*mean, s=65, c='#c93b38', edgecolors='white', linewidths=1.1, zorder=5)
    outline = vertices[[0,1,2,0]]
    ax.plot(outline[:,0], outline[:,1], color='#24362b', lw=1.3)
    for xy, label, offset in zip(vertices, ['Class 1', 'Class 2', 'Class 3'], [(-5,-15),(5,-15),(0,7)]):
        ax.annotate(label, xy, xytext=offset, textcoords='offset points', ha='center', fontsize=11)
    ax.set_title(title, fontsize=17, pad=17)
    ax.text(.5, -.21, caption, transform=ax.transAxes, ha='center', fontsize=12)
    if a < 1:
        ax.text(.5, -.30, r'$\varepsilon=0.08$', transform=ax.transAxes, ha='center', fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(-.13,1.13)
    ax.set_ylim(-.04,1.00)
    ax.axis('off')
fig.subplots_adjust(left=.03, right=.97, top=.84, bottom=.23, wspace=.22)
fig.legend([mean_dot], [r'Mean $(1/3,1/3,1/3)$'], loc='lower center',
           bbox_to_anchor=(.5, -.075), frameon=False, fontsize=12, handletextpad=.4,
           scatteryoffsets=[.5])
for ext in ['pdf','png']:
    fig.savefig(root/'figures'/f'edl_zero_evidence_baselines.{ext}', dpi=180, bbox_inches='tight', facecolor='white')
