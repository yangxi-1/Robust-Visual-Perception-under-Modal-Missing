import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# IMDb
# =========================================================
imdb_frozen = [
    0.2824, 0.2920, 0.2578, 0.3057, 0.3206,
    0.3413, 0.3923, 0.3741, 0.3651, 0.4080,
    0.3985, 0.4320, 0.4168, 0.4261, 0.4380,
    0.4113, 0.4352, 0.4491, 0.4254, 0.4426
]

imdb_unfrozen = [
    0.4757, 0.5472, 0.5685, 0.5624, 0.5605,
    0.5687, 0.5689, 0.5650, 0.5389, 0.5641,
    0.5595, 0.5681, 0.5608, 0.5590, 0.5447,
    0.5670, 0.5602, 0.5502, 0.5513, 0.5497
]

# =========================================================
# HM
# =========================================================
hm_frozen = [
    0.3622, 0.3622, 0.3753, 0.3984, 0.3904,
    0.4173, 0.4199, 0.3966, 0.4278, 0.4302,
    0.4781, 0.4688, 0.4058, 0.4204, 0.4853,
    0.4766, 0.4725, 0.4874, 0.4416, 0.4789
]

hm_unfrozen = [
    0.5281, 0.5088, 0.3763, 0.4235, 0.4324,
    0.4383, 0.4461, 0.4526, 0.4633, 0.4436,
    0.4654, 0.4934, 0.4773, 0.4605, 0.4612,
    0.4798, 0.4552, 0.4774, 0.4606, 0.4506
]

# =========================================================
# Plot
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# =========================================================
# IMDb subplot
# =========================================================
epochs_frozen_imdb = np.arange(1, len(imdb_frozen) + 1)
epochs_unfrozen_imdb = np.arange(1, len(imdb_unfrozen) + 1)

axes[0].plot(
    epochs_frozen_imdb,
    imdb_frozen,
    linewidth=2.3,
    marker='o',
    markersize=5,
    label='Frozen Encoder'
)

axes[0].plot(
    epochs_unfrozen_imdb,
    imdb_unfrozen,
    linewidth=2.3,
    marker='s',
    markersize=5,
    label='Trainable Encoder'
)

# Mark best points
best_frozen_idx = np.argmax(imdb_frozen)
best_unfrozen_idx = np.argmax(imdb_unfrozen)

axes[0].scatter(
    best_frozen_idx + 1,
    imdb_frozen[best_frozen_idx],
    s=80
)

axes[0].scatter(
    best_unfrozen_idx + 1,
    imdb_unfrozen[best_unfrozen_idx],
    s=80
)

axes[0].set_title('MM-IMDb', fontsize=15)
axes[0].set_xlabel('Epoch', fontsize=13)
axes[0].set_ylabel('Micro-F1', fontsize=13)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(fontsize=11)

# =========================================================
# HM subplot
# =========================================================
epochs_frozen_hm = np.arange(1, len(hm_frozen) + 1)
epochs_unfrozen_hm = np.arange(1, len(hm_unfrozen) + 1)

axes[1].plot(
    epochs_frozen_hm,
    hm_frozen,
    linewidth=2.3,
    marker='o',
    markersize=5,
    label='Frozen Encoder'
)

axes[1].plot(
    epochs_unfrozen_hm,
    hm_unfrozen,
    linewidth=2.3,
    marker='s',
    markersize=5,
    label='Trainable Encoder'
)

best_frozen_idx_hm = np.argmax(hm_frozen)
best_unfrozen_idx_hm = np.argmax(hm_unfrozen)

axes[1].scatter(
    best_frozen_idx_hm + 1,
    hm_frozen[best_frozen_idx_hm],
    s=80
)

axes[1].scatter(
    best_unfrozen_idx_hm + 1,
    hm_unfrozen[best_unfrozen_idx_hm],
    s=80
)

axes[1].set_title('Hateful Memes', fontsize=15)
axes[1].set_xlabel('Epoch', fontsize=13)
axes[1].set_ylabel('F1 Score', fontsize=13)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend(fontsize=11)

# =========================================================
# Global title
# =========================================================
plt.suptitle(
    'Effect of Freezing Encoders During Training',
    fontsize=17
)

plt.tight_layout()

# Save
plt.savefig('encoder_freeze_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('encoder_freeze_comparison.pdf', bbox_inches='tight')

plt.show()
