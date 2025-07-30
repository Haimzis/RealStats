import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Mean AUC values
auc_means = {
    "Ours": 0.741,
    "RIGID": 0.769,
    "AEROBLADE": 0.552,
    "ManifoldBias": 0.731
}

# Std deviations
auc_stds = {
    "Ours": 0.129,
    "RIGID": 0.151,
    "AEROBLADE": 0.236,
    "ManifoldBias": 0.175
}

# Define bar colors
colors = {
    "Ours": "#7EB6D7",          # Light blue
    "RIGID": "#F4A261",         # Orange
    "AEROBLADE": "#F7C6C7",     # Light pink
    "ManifoldBias": "#C6B4E6"   # Light purple
}

# Plot setup
plt.figure(figsize=(6, 4))
methods = list(auc_means.keys())
x_pos = np.arange(len(methods))
mean_vals = [auc_means[m] for m in methods]
std_vals = [auc_stds[m] for m in methods]
bar_colors = [colors[m] for m in methods]

# Create bar plot
bars = plt.bar(x_pos, mean_vals, yerr=std_vals, capsize=5, color=bar_colors)

# Style
plt.xticks(x_pos, methods, fontsize=11)
plt.ylabel('AUC Score', fontsize=10)
plt.ylim(0, 1.05)

# Show all spines for a square frame
for spine in plt.gca().spines.values():
    spine.set_visible(True)

plt.grid(False)

# Save figure
plt.tight_layout()
plt.savefig('auc_bar_chart_square_frame.svg', format='svg')
plt.show()
