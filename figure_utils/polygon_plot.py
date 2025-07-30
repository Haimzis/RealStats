import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "Dataset": [
        "ProGAN", "StyleGAN", "StyleGAN2", "BigGAN", "GauGAN", "CycleGAN", "StarGAN",
        "CRN", "IMLE", "SAN", "SeeingDark", "DeepFake", "SDv2", "SDXL",
        "Midjourney", "DALLE", "SDv1.4", "GLIDE_100_10", "GLIDE_100_27", "LDM"
    ],
    "RealStats-MinP (Ours)": [
        0.783, 0.697, 0.732, 0.832, 0.822, 0.923, 0.662,
        0.860, 0.857, 0.692, 0.516, 0.584, 0.672, 0.759,
        0.661, 0.700, 0.536, 0.962, 0.941, 0.632
    ],
    "RIGID": [
        0.937, 0.541, 0.633, 0.814, 0.921, 0.889, 0.620,
        0.802, 0.835, 0.743, 0.440, 0.771, 0.800, 0.854,
        0.716, 0.650, 0.592, 0.990, 0.988, 0.855
    ],
    "AeroBlade": [
        0.375, 0.303, 0.283, 0.351, 0.325, 0.180, 0.400,
        0.717, 0.719, 0.767, 0.467, 0.600, 0.237, 0.578,
        0.650, 0.761, 0.634, 0.943, 0.920, 0.841
    ],
    "ManifoldBias": [
        0.959, 0.641, 0.611, 0.862, 0.979, 0.893, 0.541,
        0.892, 0.910, 0.642, 0.760, 0.637, 0.832, 0.642,
        0.473, 0.364, 0.636, 0.887, 0.877, 0.584
    ]
}


df = pd.DataFrame(data)
categories = df["Dataset"].tolist()
methods = list(df.columns[1:])
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

colors = {
    "RealStats-MinP (Ours)": "#7EB6D7",       # Light blue
    "RIGID": "#F4A261",       # Orange
    "AeroBlade": "#F7C6C7",          # Light pink
    "ManifoldBias": "#C6B4E6"       # Light purple
}

# Plot
fig, ax = plt.subplots(figsize=(10, 6.5), subplot_kw=dict(polar=True))

# Plot each method
for method in methods:
    values = df[method].tolist()
    values += values[:1]  # Close the polygon
    ax.plot(angles, values, label=method, linewidth=2.5, color=colors[method])

# Axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=14)

# Custom radial labels at an angled offset (e.g., 120 degrees)
yticks = np.linspace(0.2, 1.0, 5)
ax.set_yticks(yticks)
ax.set_yticklabels([])  # Remove default labels

# Add custom radial labels manually at angle (e.g., 120 degrees)
label_angle_rad = np.deg2rad(45)  # 120 degrees in radians
for r in yticks:
    ax.text(label_angle_rad, r, f"{r:.1f}", fontsize=13,
            ha='center', va='center', rotation=45,
            rotation_mode='anchor')


# Style
ax.set_ylim(0, 1.0)
ax.set_rlabel_position(0)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
ax.set_aspect(1.0)  # elliptical shape

ax.yaxis.grid(True, alpha=0.2)
ax.xaxis.grid(True, alpha=0.2)

# Legend
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.99), fontsize=13)

# Layout
plt.tight_layout()
plt.savefig("polygon_plot_auc_elliptical.svg", dpi=300)
plt.show()
