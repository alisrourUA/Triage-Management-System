import os
import json
import matplotlib.pyplot as plt

REPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../report"))
os.makedirs(REPORT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-colorblind")
plt.rcParams.update({
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

#after the data changes, meaning when wee add anew patient, the statictis and visual vary, so the
# plots will stay on the old un-updated data, thus this function deletes the plots when the number
#of patients changes from the folder "reports" then generates new one according to current data
def clear_old_plots(plot_type=None, current_patient_count=None):
    meta_path = os.path.join(REPORT_DIR, "plot_meta.json")
    data = {}

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                data = json.load(f)
        except:
            data = {}

    old_count = data.get("patient_count", -1)

    # if the data did not change, keep them
    if current_patient_count is not None and old_count == current_patient_count:
        return

    # here we delete only same-type plots
    if plot_type:
        for f in os.listdir(REPORT_DIR):
            if f.endswith(".png") and plot_type in f:
                os.remove(os.path.join(REPORT_DIR, f))

    # updating data
    if current_patient_count is not None:
        data["patient_count"] = current_patient_count
        with open(meta_path, "w") as f:
            json.dump(data, f)


def compute_statistics(patients):
    priority_counts = {}
    injury_counts = {}
    conscious_counts = {}

    for p in patients:
        # Priority
        pr = int(float(p.get("Priority", 0)))
        priority_counts[pr] = priority_counts.get(pr, 0) + 1

        # Consciousness
        cns = str(p.get("Consciousness", "Unknown")).strip().capitalize()
        conscious_counts[cns] = conscious_counts.get(cns, 0) + 1

        # Injury Type 
        injury = str(p.get("Injury_Type", "None")).strip().lower()
        if injury in ("", "-", "_", "unknown", "nan", "na", "n/a", "none", "0"):
            injury = "none"
        injury = injury.capitalize()
        injury_counts[injury] = injury_counts.get(injury, 0) + 1

   #merging duplicates underr same categroy likee injury tyoe
    normalized = {}
    for k, v in injury_counts.items():
        clean_key = str(k).strip().capitalize()
        normalized[clean_key] = normalized.get(clean_key, 0) + v

    return priority_counts, normalized, conscious_counts



def plot_priority_distribution(priority_counts, current_patient_count=None):
    clear_old_plots("priority_distribution", current_patient_count)

    labels = [f"P{p}" for p in sorted(priority_counts.keys())]
    values = [priority_counts[p] for p in sorted(priority_counts.keys())]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#D9534F", "#F0AD4E", "#5BC0DE", "#5CB85C"])
    ax.set_title("Patients per Priority Level", pad=15)
    ax.set_xlabel("Priority Level (1 = High, 4 = Low)")
    ax.set_ylabel("Number of Patients")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{int(height)}",
                ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "priority_distribution.png"))
    plt.show()
    plt.close(fig)


def plot_injury_distribution(injury_counts, current_patient_count=None):
    clear_old_plots("injury_distribution", current_patient_count)

    labels = list(injury_counts.keys())
    values = list(injury_counts.values())

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%",
        startangle=90, pctdistance=0.8, colors=plt.cm.Paired.colors
    )
    ax.set_title("Injury Type Distribution", pad=20)
    for t in autotexts:
        t.set_color("black")
        t.set_fontsize(10)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "injury_distribution.png"))
    plt.show()
    plt.close(fig)


def plot_consciousness_distribution(conscious_counts, current_patient_count=None):
    clear_old_plots("consciousness_distribution", current_patient_count)

    labels = list(conscious_counts.keys())
    values = list(conscious_counts.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#5BC0DE", "#D9534F"])
    ax.set_title("Conscious vs Unconscious Patients", pad=15)
    ax.set_xlabel("State")
    ax.set_ylabel("Count")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{int(height)}",
                ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "consciousness_distribution.png"))
    plt.show()
    plt.close(fig)
