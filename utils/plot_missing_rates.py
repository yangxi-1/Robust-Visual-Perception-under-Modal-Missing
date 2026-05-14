import matplotlib.pyplot as plt

# =========================
# MM-IMDb
# =========================
drop_prob_imdb = [0.0, 0.2, 0.4, 0.6, 0.8]

imdb_with_prompt = [
    0.6161,
    0.5850,
    0.5348,
    0.4661,
    0.3088
]

imdb_without_prompt = [
    0.4193,
    0.5957,
    0.5319,
    0.3674,
    0.3226
]

# =========================
# Hateful Memes
# =========================
drop_prob_hm = [0.0, 0.2, 0.4, 0.6, 0.8]

hm_with_prompt = [
    0.4888,
    0.5144,
    0.4565,
    0.4352,
    0.3648
]

hm_without_prompt = [
    0.4771,
    0.4945,
    0.4610,
    0.4217,
    0.3683
]

# =========================
# Create Figure
# =========================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# =====================================
# Left subplot: MM-IMDb
# =====================================
axes[0].plot(
    drop_prob_imdb,
    imdb_with_prompt,
    marker='o',
    linewidth=2.5,
    markersize=7,
    label='With Prompts'
)

axes[0].plot(
    drop_prob_imdb,
    imdb_without_prompt,
    marker='s',
    linewidth=2.5,
    markersize=7,
    label='Without Prompts'
)

axes[0].set_title('MM-IMDb', fontsize=15)
axes[0].set_xlabel('Missing Rate', fontsize=13)
axes[0].set_ylabel('Micro-F1', fontsize=13)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(fontsize=11)

# =====================================
# Right subplot: Hateful Memes
# =====================================
axes[1].plot(
    drop_prob_hm,
    hm_with_prompt,
    marker='o',
    linewidth=2.5,
    markersize=7,
    label='With Prompts'
)

axes[1].plot(
    drop_prob_hm,
    hm_without_prompt,
    marker='s',
    linewidth=2.5,
    markersize=7,
    label='Without Prompts'
)

axes[1].set_title('Hateful Memes', fontsize=15)
axes[1].set_xlabel('Missing Rate', fontsize=13)
axes[1].set_ylabel('Micro-F1', fontsize=13)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend(fontsize=11)

# =====================================
# Global layout
# =====================================
plt.suptitle(
    'Effect of Prompts under Different Missing Rates',
    fontsize=17
)

plt.tight_layout()

# Save
plt.savefig('prompt_comparison_all.png', dpi=300, bbox_inches='tight')
plt.savefig('prompt_comparison_all.pdf', bbox_inches='tight')

# Show
plt.show()
