import matplotlib.pyplot as plt
import numpy as np

# Font size hyperparameters
title_fontsize = 16
label_fontsize = 14
tick_fontsize = 12
legend_fontsize = 12

# ========== Peak GPU Memory Plot ==========
methods = [
    'Ours (1 worker)', 'Ours (2 workers)', 'Ours (4 workers)',
    'RIGID (BS = 128)', 'ManifoldBias (BS = 1)', 'AEROBLADE (BS = 128)'
]
memories = [7, 11, 22, 6, 40, 75]
colors = {
    "Ours": "#7EB6D7", "RIGID": "#F4A261",
    "AEROBLADE": "#F7C6C7", "ManifoldBias": "#C6B4E6"
}
bar_colors = [
    colors["Ours"], colors["Ours"], colors["Ours"],
    colors["RIGID"], colors["ManifoldBias"], colors["AEROBLADE"]
]

plt.figure(figsize=(8, 4))
plt.bar(methods, memories, color=bar_colors)
plt.ylabel("Peak GPU Memory Usage (GB)", fontsize=label_fontsize)
plt.title("Peak GPU Memory Usage per Method", fontsize=title_fontsize)
plt.xticks(rotation=30, fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("peak_gpu_memory_usage.png")
plt.show()

# ========== Independent Subset Runtime Plot ==========
stat_counts = np.array([4, 8, 16, 32])
graph_times = np.array([7.31, 12.29, 34.8, 36.52])   # ms
chi2_times = np.array([17.15, 46.91, 163.15, 174.014])    # ms

plt.figure(figsize=(8, 4))
plt.plot(stat_counts, graph_times, label="Graph Construction + Max Clique", marker='o')
plt.plot(stat_counts, chi2_times, label="Chi² Pairwise Computation", marker='s')

plt.xlabel("Number of Statistics", fontsize=label_fontsize)
plt.ylabel("Runtime (ms)", fontsize=label_fontsize)
plt.title("Independent Subset Selection Runtime vs Number of Statistics", fontsize=title_fontsize)
plt.xticks(stat_counts, fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig("clique_extraction_runtime_analysis.png")
plt.show()

# ========== Statistic Extraction Runtime vs Workers ==========
graph_times_1w = np.array([307937.03, 583569.25, 933013.23, np.nan]) / 1000 / 60
graph_times_2w = np.array([177185.02, 364346.16, 574692.88, np.nan]) / 1000 / 60
graph_times_4w = np.array([138118.98, 284111.58, 452862.66, 976152.55]) / 1000 / 60

plt.figure(figsize=(8, 4))
plt.plot(stat_counts, graph_times_1w, label="1 GPU Worker", marker='o')
plt.plot(stat_counts, graph_times_2w, label="2 GPU Workers", marker='s')
plt.plot(stat_counts, graph_times_4w, label="4 GPU Workers", marker='^')

plt.xlabel("Number of Statistics", fontsize=label_fontsize)
plt.ylabel("Runtime (min)", fontsize=label_fontsize)
plt.title("Statistic Extraction Runtime vs Number of Statistics", fontsize=title_fontsize)
plt.xticks(stat_counts, fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig("statistic_extraction_runtime_by_workers.png")
plt.show()
