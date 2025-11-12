# this file is designed for handling the dataset, loading, saving, cleaning, and preparing patient data.


import os
import csv
from src.triage_logic import assign_priority
from src.regression import predict_recovery_time, train_recovery_model

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))
MAIN_DATASET = os.path.join(DATA_DIR, "patients.csv")
os.makedirs(DATA_DIR, exist_ok=True)

FIELDS = [
    "Name", "Age", "Heart_Rate", "Consciousness", "Injury_Type",
    "Blood_Pressure", "Oxygen_Level", "Recovery_Time", "Priority"
]


def _num_or_default(value, default="0"):
    try:
        return str(float(value))
    except:
        return default

def _cap(value, default="Unknown"):
    v = str(value).strip()
    return v.capitalize() if v else default

def _normalize_injury(v: str) -> str:
    v = str(v).strip().lower()
    if v in ("", "-", "_", "unknown", "nan", "na", "n/a", "none", "0"):
        return "None"
    return v.capitalize().strip()


def clean_record(p):
    """Normalize one patient record to match schema and correct outliers."""
    c = {}
    c["Name"] = str(p.get("Name", "")).strip().title() or "Unknown"

    c["Age"] = _num_or_default(p.get("Age", "0"))
    c["Heart_Rate"] = _num_or_default(p.get("Heart_Rate", "0"))
    c["Blood_Pressure"] = _num_or_default(p.get("Blood_Pressure", "0"))
    c["Oxygen_Level"] = _num_or_default(p.get("Oxygen_Level", "0"))
    c["Recovery_Time"] = _num_or_default(p.get("Recovery_Time", "0"))
    c["Priority"] = _num_or_default(p.get("Priority", "0"))

    
    state = str(p.get("Consciousness", "")).strip().lower()
    if state in ("", "none", "unknown", "nan", "-", "_"):
        c["Consciousness"] = "Conscious"  # default
    elif "un" in state and "conscious" in state:
        c["Consciousness"] = "Unconscious"
    else:
        c["Consciousness"] = "Conscious"

    injury = str(p.get("Injury_Type", "Unknown")).strip().capitalize()
    if injury in ("", "-", "Unknown", "Nan", "N/A", "None"):
        injury = "None"
    c["Injury_Type"] = injury

    try:
        hr = float(c["Heart_Rate"])
        if hr < 30 or hr > 200:  # normal heart rate range for humans
            c["Heart_Rate"] = "100"
    except:
        c["Heart_Rate"] = "100"

    try:
        o2 = float(c["Oxygen_Level"])
        if o2 < 70 or o2 > 100:  # normal human oxygen range
            c["Oxygen_Level"] = "95"
    except:
        c["Oxygen_Level"] = "95"

    try:
        bp = float(c["Blood_Pressure"])
        if bp < 40 or bp > 180:  # normal blood pressure range for humans
            c["Blood_Pressure"] = "120"
    except:
        c["Blood_Pressure"] = "120"

    return c


def clean_dataset(patients):
    return [clean_record(p) for p in patients]

def load_csv(path=MAIN_DATASET):
    if not os.path.exists(path):
        print(f"[WARN] Dataset not found: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = []
        for row in reader:
            if not any(row.values()):
                continue
            data.append(clean_record(row))

    save_csv(path, data)
    print(f"[LOAD] Loaded and cleaned {len(data)} records from {path}")
    return data

def save_csv(path, data, fieldnames=FIELDS):
    cleaned_data = [clean_record(p) for p in data]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_data)
    print(f"[SAVE] Saved {len(cleaned_data)} records → {path}")


def preprocess_dataset(patients):
   
    valid = [p for p in patients if float(p.get("Recovery_Time", 0)) > 0]
    if valid:
        train_recovery_model(valid)

    # now, we compute priority and fill missing recovery times
    for p in patients:
        #here we assign the priority according the triage rules we assigned.
        try:
            p["Priority"] = assign_priority(p)
        except Exception:
            p["Priority"] = 0

        # this part predicts recovery time if its value is missing or it is zero
        try:
            if not p.get("Recovery_Time") or float(p["Recovery_Time"]) == 0:
                p["Recovery_Time"] = round(predict_recovery_time(p), 2)
        except Exception:
            p["Recovery_Time"] = "0"

    return patients
