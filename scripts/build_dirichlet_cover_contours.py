"""Title illustration using the thesis Dirichlet density, eq:dirichlet-density.
Asymmetric alpha=(2,4,2). No experimental results.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import gammaln

root = Path(__file__).resolve().parents[1]
resolution = 320
pi = np.array([(i/resolution, j/resolution, 1-(i+j)/resolution)
               for i in range(resolution+1) for j in range(resolution+1-i)])
alpha = np.array([2., 4., 2.])
x = pi[:, 1] + 0.5*pi[:, 2]
y = np.sqrt(3)/2*pi[:, 2]
density = np.zeros(len(pi))
inside = (pi > 0).all(axis=1)
density[inside] = np.exp(gammaln(alpha.sum()) - gammaln(alpha).sum()
                          + ((alpha-1)*np.log(pi[inside])).sum(axis=1))
tri = mtri.Triangulation(x, y)
# White, tints of LMU green (#00883a), and a darker shade at the peak.
cmap = LinearSegmentedColormap.from_list('lmu_density',
    ['#f8fcf9', '#d8eddf', '#a6d5ba', '#6cba8d', '#359f63', '#00883a', '#005525'])
fig, ax = plt.subplots(figsize=(6, 5.23))
fig.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.985)
ax.tricontourf(tri, density, levels=np.linspace(0, density.max()*1.000001, 100), cmap=cmap)
ax.tricontour(tri, density, levels=np.linspace(0, density.max(), 10)[1:-1],
              colors='#00632b', linewidths=0.8, alpha=0.30)
ax.plot([0, 1, .5, 0], [0, 0, np.sqrt(3)/2, 0], color='#007333', linewidth=1.4)
ax.set_aspect('equal')
ax.set_xlim(-.014, 1.014)
ax.set_ylim(-.014, np.sqrt(3)/2+.014)
ax.axis('off')
for extension in ('pdf', 'png'):
    fig.savefig(root/'figures'/f'dirichlet_cover_contours.{extension}', dpi=320,
                bbox_inches='tight', pad_inches=0.015, transparent=True)
plt.close(fig)
