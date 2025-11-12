import os
from src.admissions_io import load_csv, clean_dataset, preprocess_dataset, MAIN_DATASET, save_csv
from src.console_admit import run_admission_session
from src.triage_logic import assign_priority
from src.nb_priority import nb_train, nb_predict
from src.queue_manager import sort_by_priority
from src.kmeans import kmeans
from src.visualize_clusters import plot_clusters
from src.statistics_visuals import (
    compute_statistics,
    plot_priority_distribution,
    plot_injury_distribution,
    plot_consciousness_distribution,
)


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")

def pause():
    input("\nPress Enter to continue...")


def show_patients_table(patients, limit=20):
    print("\n{:<18} {:>3}  {:>4} {:>5} {:>5} {:<12} {:<10} {:>4} {:>7}".format(
        "Name", "Age", "HR", "BP", "O2", "State", "Injury", "Prio", "Recov"
    ))
    print("-" * 90)
    for p in patients[:limit]:
        print("{:<18} {:>3}  {:>4} {:>5} {:>5} {:<12} {:<10} {:>4} {:>7}".format(
            str(p.get("Name", ""))[:18],
            int(float(p.get("Age", 0))),
            int(float(p.get("Heart_Rate", 0))),
            int(float(p.get("Blood_Pressure", 0))),
            int(float(p.get("Oxygen_Level", 0))),
            str(p.get("Consciousness", ""))[:12],
            str(p.get("Injury_Type", ""))[:10],
            int(float(p.get("Priority", 0))),
            str(p.get("Recovery_Time", 0))[:7],
        ))
    if len(patients) > limit:
        print(f"... ({len(patients) - limit} more)")


def filter_by_injury_menu(patients):
    counts = {}
    for p in patients:
        key = str(p.get("Injury_Type", "Unknown")).capitalize()
        counts[key] = counts.get(key, 0) + 1
    options = sorted(counts.items(), key=lambda kv: kv[0])
    print("\nAvailable injury types:")
    for i, (k, v) in enumerate(options, start=1):
        print(f"{i}. {k} ({v})")
    print(f"0. All injuries (total {len(patients)})")

    choice = input(f"\nChoose an injury index (0-{len(options)}): ").strip()
    if choice == "0":
        return patients
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            wanted = options[idx][0]
            return [p for p in patients if str(p.get("Injury_Type", "")).capitalize() == wanted]
    except:
        pass
    return patients


def main():
    
    raw = load_csv(MAIN_DATASET)
    patients = clean_dataset(raw)
    patients = preprocess_dataset(patients)

    while True:
        clear_console()
        print("=" * 70)
        print("TRIAGE MANAGEMENT SYSTEM DASHBOARD (in-memory)".center(70))
        print("=" * 70)
        print("1) Add a new patient")
        print("2) View all patients")
        print("3) View statistics & visualizations")
        print("4) View patients sorted by PRIORITY")
        print("5) View patients filtered by INJURY TYPE")
        print("6) View K-Means CLUSTERS (HR vs O₂)")
        print("0) Exit")
        print("=" * 70)

        choice = input("Choose an option: ").strip()

        
        if choice == "1":
            new_patients = run_admission_session(assign_priority, nb_train, nb_predict, patients)
            if new_patients:
                patients.extend(new_patients)
                print(f"{len(new_patients)} patients added successfully.")
            pause()

       
        elif choice == "2":
            show_patients_table(patients, limit=40)
            pause()

        
        elif choice == "3":
            prio, injury, consci = compute_statistics(patients)
            current_count = len(patients)
            print("\n--- UPDATED STATISTICS ---")
            print("Priority Counts:", prio)
            print("Injury Type Counts:", injury)
            print("Conscious/Unconscious Counts:", consci)

            while True:
                print("\nWhich visualization would you like to see?")
                print("1) Priority Distribution")
                print("2) Injury Type Distribution")
                print("3) Consciousness Distribution")
                print("4) Show ALL")
                print("0) Back to Main Menu")

                sub_choice = input("Enter choice: ").strip()

                if sub_choice == "1":
                    plot_priority_distribution(prio, current_patient_count=current_count)
                    print("Priority distribution updated and saved.")
                elif sub_choice == "2":
                    plot_injury_distribution(injury, current_patient_count=current_count)
                    print("Injury distribution updated and saved.")
                elif sub_choice == "3":
                    plot_consciousness_distribution(consci, current_patient_count=current_count)
                    print("Consciousness distribution updated and saved.")
                elif sub_choice == "4":
                    plot_priority_distribution(prio, current_patient_count=current_count)
                    plot_injury_distribution(injury, current_patient_count=current_count)
                    plot_consciousness_distribution(consci, current_patient_count=current_count)
                    print("All charts updated and saved.")
                elif sub_choice == "0":
                    break
                else:
                    print("Invalid choice.")
            pause()

        
        elif choice == "4":
            sorted_list = sort_by_priority(list(patients))
            print("\nPatients sorted by priority:")
            show_patients_table(sorted_list, limit=40)
            pause()

        
        elif choice == "5":
            filtered = filter_by_injury_menu(patients)
            print("\nPatients (filtered):")
            show_patients_table(filtered, limit=40)
            pause()

        
        elif choice == "6":
            pts = []
            for p in patients:
                try:
                    pts.append((float(p["Heart_Rate"]), float(p["Oxygen_Level"])))
                except:
                    pass

            if len(pts) < 3:
                print("Not enough numeric data for clustering (need ≥3).")
            else:
                centroids, clusters = kmeans(pts, k=3)
                print("\nK-Means clusters:")
                for i, c in enumerate(clusters):
                    print(f"  Cluster {i+1}: {len(c)} points | centroid={centroids[i]}")
                plot_clusters(clusters, centroids, current_patient_count=len(patients))
            pause()

       
        elif choice == "0":
            save_csv(MAIN_DATASET, patients)
            print("\nAll patients saved. Goodbye.")
            break

        else:
            print("Invalid choice.")
            pause()

if __name__ == "__main__":
    main()
