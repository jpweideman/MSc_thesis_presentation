"""Title illustration from the thesis Dirichlet density (eq:dirichlet-density).

A three-class Dirichlet with alpha=(3,5,7). Surface height is its density.
This is a mathematical illustration, not an experimental result.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import gammaln
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
root=Path(__file__).resolve().parents[1]
alpha=np.array([3.,5.,7.])
N=135
pi=np.array([(i/N,j/N,1-(i+j)/N) for i in range(N+1) for j in range(N+1-i)])
x=pi[:,1]+.5*pi[:,2]
y=np.sqrt(3)/2*pi[:,2]
z=np.zeros(len(pi))
interior=(pi>0).all(axis=1)
z[interior]=np.exp(gammaln(alpha.sum())-gammaln(alpha).sum()+((alpha-1)*np.log(pi[interior])).sum(axis=1))
tri=mtri.Triangulation(x,y)
cmap=LinearSegmentedColormap.from_list('cover',['#edf5f2','#b3e0cf','#4cbba1','#00883a','#006b58','#003f47'])
fig=plt.figure(figsize=(6.8,5.5),facecolor='white')
ax=fig.add_axes([-.03,-.04,1.06,1.10],projection='3d')
base=-z.max()*.075
verts=[(0,0,base),(1,0,base),(.5,np.sqrt(3)/2,base)]
ax.add_collection3d(Poly3DCollection([verts],facecolor='#edf4f1',edgecolor='#3a786b',linewidth=1.0))
ax.tricontour(tri,z,levels=np.linspace(z.max()*.08,z.max()*.9,9),zdir='z',offset=base,colors='#549a87',linewidths=.65,alpha=.6)
ax.plot_trisurf(tri,z,cmap=cmap,linewidth=0,antialiased=True,shade=True,rasterized=True)
ax.tricontour(tri,z,levels=np.linspace(z.max()*.15,z.max()*.96,10),colors='#d4ede3',linewidths=.52,alpha=.65)
ax.view_init(elev=29,azim=-66)
ax.set_box_aspect((1,.87,.56))
ax.set(xlim=(-.01,1.01),ylim=(-.02,.89),zlim=(base,z.max()*1.04))
ax.set_axis_off()
# Crop the plotting canvas to the visible illustration.
from matplotlib.transforms import Bbox
fig.canvas.draw()
rgba = np.asarray(fig.canvas.buffer_rgba())
rows, cols = np.where(np.any(rgba[:, :, :3] < 245, axis=2))
h = rgba.shape[0]
pad = 12
crop = Bbox.from_extents((cols.min()-pad)/fig.dpi,
                         (h-rows.max()-pad)/fig.dpi,
                         (cols.max()+pad)/fig.dpi,
                         (h-rows.min()+pad)/fig.dpi)
for ext in ('pdf','png'):
    fig.savefig(root/'figures'/f'dirichlet_cover_surface.{ext}',dpi=350,bbox_inches=crop,pad_inches=.02,facecolor='white')
plt.close(fig)
print('Created a Dirichlet density surface with contour lines and a triangular base.')
