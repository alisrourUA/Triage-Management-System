# this file is responsible for admitting patients directly into the in-memory list

from .nb_priority import MIN_SAMPLES_FOR_ML
from .regression import fit_linear_regression, predict_one

def _ask_int(prompt, default=0):
    s = input(prompt).strip()
    if not s:
        return default
    try:
        return int(float(s))
    except:
        return default

def run_admission_session(assign_priority_fn, nb_train_fn=None, nb_predict_fn=None, patients=None):
    print("=" * 70)
    print(" ADMISSION DASHBOARD ".center(70, "="))
    print("=" * 70)

    new_patients = []

    while True:
        print("\nNew patient entry:")
        name = input("Name: ").strip().title() or "Unknown"
        age = _ask_int("Age: ")
        hr = _ask_int("Heart Rate (bpm): ")
        bp = _ask_int("Blood Pressure: ")
        o2 = _ask_int("Oxygen Level: ")
        conscious = input("Is conscious? (y/n): ").strip().lower()
        state = "Conscious" if conscious == "y" else "Unconscious"
        injury = input("Injury (minor/bleeding/fracture/burn/none): ").strip().capitalize() or "Unknown"

        #prior pred
        rule_priority = assign_priority_fn({
            "Age": age, "Heart_Rate": hr, "Blood_Pressure": bp,
            "Oxygen_Level": o2, "Consciousness": state, "Injury_Type": injury
        })

        ml_priority = None
        if patients and nb_train_fn and nb_predict_fn and len(patients) >= MIN_SAMPLES_FOR_ML:
            nb_model = nb_train_fn(patients)
            ml_priority = nb_predict_fn(nb_model, {
                "Age": age, "Heart_Rate": hr, "Blood_Pressure": bp,
                "Oxygen_Level": o2, "Consciousness": state, "Injury_Type": injury
            })

        #recov pred
        recovery_time = 0.0
        if patients:
            features = ["Heart_Rate", "Age", "Blood_Pressure", "Oxygen_Level"]
            reg_model = fit_linear_regression(patients, features)
            recovery_time = round(predict_one(reg_model, {
                "Age": age, "Heart_Rate": hr, "Blood_Pressure": bp, "Oxygen_Level": o2
            }), 2)

        final_priority = rule_priority

        print("\n----------------------------")
        print(f"ML Predicted Priority: {ml_priority if ml_priority else 'N/A'}")
        print(f"Rule-based Priority:   {rule_priority}")
        print(f"Predicted Recovery:    {recovery_time} days")
        print(f"Final Assigned:        {final_priority}")
        print("----------------------------")

        patient = {
            "Name": name,
            "Age": age,
            "Heart_Rate": hr,
            "Blood_Pressure": bp,
            "Oxygen_Level": o2,
            "Consciousness": state,
            "Injury_Type": injury,
            "Priority": final_priority,
            "Recovery_Time": recovery_time
        }
        new_patients.append(patient)

        again = input("Add another? (y/n): ").strip().lower()
        if again != "y":
            print("\nSession complete. All patients stored in memory!\n")
            break

    return new_patients
