# this file is the central analysis for the admissions system.
from src.admissions_io import (
    merge_datasets,
    load_csv,
    preprocess_dataset,
    dataset_to_dict,
    get_combined_dataset_path
)
from src.triage_logic import assign_priority
from src.regression import train_recovery_model, predict_recovery_time
from src.statistics_visuals import (
    compute_statistics,
    plot_priority_distribution,
    plot_injury_distribution,
    plot_consciousness_distribution,
)


def analyze_admissions():
    print("\n Loading unified dataset...")
    df = load_csv(get_combined_dataset_path())
    if not df:
        print(" No records found. Please add patients first.")
        return

    print(f"Loaded {len(df)} patients.")
    patients = dataset_to_dict(df)

    print("\n Training recovery-time regression model...")
    train_recovery_model(patients)

    print("Recomputing derived attributes...")
    for p in patients:
        p["Priority"] = assign_priority(p)
        try:
            p["Recovery_Time"] = round(predict_recovery_time(p), 2)
        except Exception:
            pass

    print("\nComputing updated statistics...")
    priority_counts, injury_counts, conscious_counts = compute_statistics(patients)

    print("\n--- UPDATED STATISTICS ---")
    print("Priority Counts:", priority_counts)
    print("Injury Type Counts:", injury_counts)
    print("Conscious/Unconscious Counts:", conscious_counts)

    print("\nGenerating updated charts...")
    plot_priority_distribution(priority_counts)
    plot_injury_distribution(injury_counts)
    plot_consciousness_distribution(conscious_counts)

    

    print("\n Analysis complete, the data and visuals updated.")
