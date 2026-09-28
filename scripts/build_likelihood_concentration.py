"""Likelihood against total concentration for one CIFAR-10 test image.

Controlled likelihood illustration, not a model comparison and not training.
The weights of the trained DM-eBNN are held at one saved posterior sample w, so
the mean vector m(x;w) is fixed. The same fixed mean vector is used for both
observation models. Only the total concentration alpha_0 varies, with
alpha = alpha_0 m (eq:mean-total-concentration-parameterisation).

Single label. eq:dm-categorical-limit gives the likelihood m_y, the same for
every alpha_0 (eq:ebnn-likelihood-invariance).
Count vector. eq:dm-mass changes with alpha_0 when R > 1
(sec:dm-ebnn-concentration-information).

Every alpha_c(x;w) = e_c(x;w) + 1 > 1 (eq:edl-dirichlet-concentration). At a
fixed mean vector the network therefore reaches only alpha_0 > 1 / min_c m_c.
The x axis starts at that value, so every plotted point is reachable.
The y axis shows the likelihood itself and starts at zero. The single label
axis runs from 0 to 1. The count likelihood is about 1e-6, the probability of
one exact count vector, so its axis states the scale factor in the label,
Likelihood (x 10^-7). The count figure uses two panels because the two scales
differ by about a million.

Selection. Image 6505 is the only held-out CIFAR-10H image whose count
likelihood has a clear maximum inside the reachable range, across all saved
DM-eBNN runs and weight samples. For most images the count likelihood falls
from the smallest reachable alpha_0. The example is not representative.

This evaluates the thesis likelihoods at a fixed mean vector. It is not a new
experimental result. The script needs numpy, matplotlib and Pillow and writes
only inside the presentation folder.
"""
from pathlib import Path
from math import lgamma
import argparse
import hashlib
import json
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--experiments', type=Path, default=root.parent / 'BNN-EDL')
args = parser.parse_args()
figures = root / 'figures'
BLUE, GREEN = '#326a9f', '#00883a'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.labelsize': 13,
    'axes.titlesize': 13, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#aaaaaa', 'xtick.color': '#444444', 'ytick.color': '#444444',
    'legend.frameon': False, 'savefig.facecolor': 'white', 'pdf.fonttype': 42,
})
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
IMAGE = 6505
DM_RUN, DM_SAMPLE = 'e2_c10h_ebnn_dm_mode50_warm_s0', 6  # DM-eBNN, CIFAR-10H, prior mode 50, saved sample 6 of 40
dm_path = args.experiments / 'outputs' / DM_RUN / 'arrays/cifar10h_test.npz'
counts_path = args.experiments / 'data/cifar10h/cifar10h-counts.npy'
images_path = args.experiments / 'data/cifar-10-batches-py/test_batch'


def dm_log_mass(n, alpha):
    """Thesis eq:dm-mass for one count vector n and concentration vector alpha."""
    total, alpha_0 = int(sum(n)), float(sum(alpha))
    value = lgamma(total + 1) - sum(lgamma(k + 1) for k in n)
    value += lgamma(alpha_0) - lgamma(alpha_0 + total)
    value += sum(lgamma(a + k) - lgamma(a) for a, k in zip(alpha, n))
    return value


def sample_mean(arrays, sample, row):
    """Mean vector m(x;w) at one saved weight sample, normalised after float16 storage."""
    mean = arrays['snapshot_means'][sample, row, :].astype(np.float64)
    return mean / mean.sum()


with images_path.open('rb') as handle:
    batch = pickle.load(handle, encoding='bytes')
label = int(batch[b'labels'][IMAGE])
image = batch[b'data'][IMAGE].reshape(3, 32, 32).transpose(1, 2, 0)
Image.fromarray(image).save(figures / 'likelihood_example_input.png')
all_counts = np.load(counts_path)
dm = np.load(dm_path)
# DM-eBNN rows are local to the held-out split. Match the count vector uniquely.
dm_row = int(np.flatnonzero(np.all(dm['true_counts'] == all_counts[IMAGE], axis=1))[0])
assert np.flatnonzero(np.all(all_counts == dm['true_counts'][dm_row], axis=1)).tolist() == [IMAGE]
counts = [int(k) for k in all_counts[IMAGE]]
one_label = [int(c == label) for c in range(len(CLASSES))]
mean = sample_mean(dm, DM_SAMPLE, dm_row)


def curve(n):
    """Likelihood from the smallest reachable alpha_0, 1 / min_c m_c, to 1000."""
    totals = np.geomspace(1.0 / mean.min() * (1 + 1e-9), 1000.0, 900)
    return totals, np.exp([dm_log_mass(n, t * mean) for t in totals])


x, single = curve(one_label)
_, count = curve(counts)
# eq:dm-categorical-limit. One label gives m_y at every alpha_0.
assert np.allclose(single, mean[label])
peak = int(count.argmax())
count_rel = count / count.max()   # recorded in the sources file only


def panel(ax, y, color, title, ylabel, ylim, yticks, mark_peak=False):
    ax.plot(x, y, color=color, lw=2.8)
    if mark_peak:
        ax.plot([x[y.argmax()]], [y.max()], 'o', color=color, ms=6)
    ax.set(xscale='log', xlim=(x[0], x[-1]), ylim=ylim, yticks=yticks,
           xlabel=r'Total concentration $\alpha_0$', ylabel=ylabel, title=title)
    ax.set_xticks([50, 100, 300, 1000], ['50', '100', '300', '1000'])
    ax.minorticks_off()


def save(fig, name):
    fig.savefig(figures / f'{name}.pdf', bbox_inches='tight', pad_inches=0.08)
    fig.savefig(figures / f'{name}.png', dpi=160, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


name = CLASSES[label]
fig, ax = plt.subplots(figsize=(5.4, 1.85))
panel(ax, single, BLUE, '', 'Likelihood', (0, 1), [0, 0.5, 1])
save(fig, 'likelihood_concentration_single')

fig, axs = plt.subplots(1, 2, figsize=(9.4, 1.85), gridspec_kw={'wspace': 0.3})
panel(axs[0], single, BLUE, f'Single label, {name}', 'Likelihood', (0, 1), [0, 0.5, 1])
exponent = int(np.floor(np.log10(count.max())))
scaled = count / 10.0 ** exponent
top = 3 * np.ceil(scaled.max() / 3)
panel(axs[1], scaled, GREEN, f'{sum(counts)} CIFAR-10H labels',
      rf'Likelihood ($\times 10^{{{exponent}}}$)', (0, 1.15 * top), np.arange(0, top + 1, 3), mark_peak=True)
save(fig, 'likelihood_concentration_counts')

sources = [dm_path, counts_path, images_path]
(figures / 'likelihood_concentration_sources.json').write_text(json.dumps({
    'generator': 'scripts/build_likelihood_concentration.py',
    'kind': 'Controlled likelihood illustration at one fixed mean vector. Not a model comparison or a new experimental result.',
    'thesis_labels': ['eq:dm-mass', 'eq:dm-categorical-limit', 'eq:mean-total-concentration-parameterisation',
                      'eq:edl-dirichlet-concentration', 'eq:ebnn-likelihood-invariance',
                      'sec:dm-ebnn-concentration-information'],
    'cifar10_test_image_index': IMAGE,
    'label': name,
    'count_vector': dict(zip(CLASSES, counts)),
    'selection': ('Only held-out CIFAR-10H image whose count likelihood has a clear maximum inside the '
                  'reachable range across all saved DM-eBNN runs and weight samples. For most images the '
                  'count likelihood falls from the smallest reachable alpha_0. Not representative.'),
    'mean_vector_source': {'run': DM_RUN, 'row': dm_row, 'weight_sample': DM_SAMPLE,
                           'alpha_0_at_this_sample': round(float(dm['snapshot_alpha0'][DM_SAMPLE, dm_row]), 2)},
    'mean_vector': dict(zip(CLASSES, np.round(mean, 4).tolist())),
    'smallest_reachable_alpha_0': round(float(x[0]), 2),
    'likelihood_one_label': round(float(single[0]), 4),
    'count_likelihood_maximiser': round(float(x[peak]), 2),
    'count_likelihood_at_maximum': float(count[peak]),
    'count_relative_at_smallest_reachable': round(float(count_rel[0]), 3),
    'count_relative_at_1000': round(float(count_rel[-1]), 3),
    'x_axis': 'From the smallest reachable alpha_0 at the fixed mean vector, 1/min_c m_c, to 1000. Log scale.',
    'y_axis': 'Likelihood itself, starting at zero. Single label axis 0 to 1. Count axis labelled Likelihood (x 10^-7). Two panels because the scales differ by about 1e6.',
    'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
}, indent=2) + '\n')
print(f'Image {IMAGE}, label {name}, m_y {mean[label]:.3f}, alpha_0 from {x[0]:.1f}.')
print(f'Count likelihood peaks at {x[peak]:.1f}. Relative {count_rel[0]:.2f} at the start, {count_rel[-1]:.2f} at 1000.')
print('Counts', {c: k for c, k in zip(CLASSES, counts) if k})
