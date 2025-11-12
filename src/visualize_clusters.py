# K-Means cluster plot (persistent version)
import os
import matplotlib.pyplot as plt
from .statistics_visuals import clear_old_plots, REPORT_DIR

def plot_clusters(clusters, centroids, current_patient_count=None):
    """Visualize clusters with persistence."""
    clear_old_plots("cluster_plot", current_patient_count)

    colors = ["#FF5733", "#33C1FF", "#75FF33", "#FF33A8", "#FFD433"]
    plt.figure(figsize=(7, 5))

    for i, cluster in enumerate(clusters):
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        plt.scatter(xs, ys, color=colors[i % len(colors)], label=f"Cluster {i+1}")

    for i, c in enumerate(centroids):
        plt.scatter(c[0], c[1], color="black", marker="X", s=120, label=f"Centroid {i+1}")

    plt.title("K-Means Clustering (Heart Rate vs Oxygen Level)")
    plt.xlabel("Heart Rate (bpm)")
    plt.ylabel("Oxygen Level (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(REPORT_DIR, "cluster_plot.png")
    plt.savefig(out_path)
    plt.show()
    plt.close()

    print(f"Cluster plot saved to: {out_path}")
