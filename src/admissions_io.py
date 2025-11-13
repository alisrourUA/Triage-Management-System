import os
import csv
from src.triage_logic import assign_priority
from src.regression import predict_recovery_time, train_recovery_model

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))
MAIN_DATASET = os.path.join(DATA_DIR, "patients.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# columns that the original dataset has.
ORIGINAL_FIELDS = [
    "Name", "Age", "Heart_Rate", "Consciousness",
    "Injury_Type", "Blood_Pressure", "Oxygen_Level",
    "Recovery_Time"
]


def _num_or_default(value, default="0"):
    try:
        return str(float(value))
    except:
        return default

def _normalize_consciousness(v):
    v = str(v).strip().lower()
    if v in ("", "none", "unknown", "nan", "-", "_"):
        return "Conscious"
    if "un" in v and "conscious" in v:
        return "Unconscious"
    return "Conscious"

def _normalize_injury(v):
    v = str(v).strip().lower()
    if v in ("", "-", "_", "unknown", "nan", "na", "n/a", "none", "0"):
        return "None"
    return v.capitalize()


def clean_record(p):
    """Normalize record fields """
    c = {}

    c["Name"] = str(p.get("Name", "")).strip().title() or "Unknown"

    c["Age"] = _num_or_default(p.get("Age", "0"))
    c["Heart_Rate"] = _num_or_default(p.get("Heart_Rate", "0"))
    c["Blood_Pressure"] = _num_or_default(p.get("Blood_Pressure", "0"))
    c["Oxygen_Level"] = _num_or_default(p.get("Oxygen_Level", "0"))
    c["Recovery_Time"] = _num_or_default(p.get("Recovery_Time", "0"))

    c["Consciousness"] = _normalize_consciousness(p.get("Consciousness", "Conscious"))
    c["Injury_Type"] = _normalize_injury(p.get("Injury_Type", "None"))

    return c

def clean_dataset(patients):
    return [clean_record(p) for p in patients]


def load_csv(path=MAIN_DATASET):
    """Load the dataset"""
    if not os.path.exists(path):
        print(f"[WARN] Dataset not found: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = []

        for row in reader:
            # Ignore empty lines
            if not any(row.values()):
                continue

            # Keep only original columns
            cleaned_row = {col: row.get(col, "") for col in ORIGINAL_FIELDS}
            data.append(cleaned_row)

    print(f"[LOAD] Loaded {len(data)} records from {path}")
    return data


def save_csv(path, data):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ORIGINAL_FIELDS)
        writer.writeheader()

        for p in data:
            row = {field: p.get(field, "") for field in ORIGINAL_FIELDS}
            writer.writerow(row)

    print(f"[SAVE] Saved {len(data)} records → {path}")


def preprocess_dataset(patients):
    """
    Add priority and predict missing recovery time.
    This modifies only the in-memory dictionary.
    """

    # Train regression model only with valid recovery values
    valid = [p for p in patients if float(p.get("Recovery_Time", 0)) > 0]
    if valid:
        train_recovery_model(valid)

    for p in patients:

        try:
            p["Priority"] = assign_priority(p)
        except:
            p["Priority"] = 0

        # predict recovery time if it is missing or zero
        try:
            if not p.get("Recovery_Time") or float(p["Recovery_Time"]) == 0:
                p["Recovery_Time"] = round(predict_recovery_time(p), 2)
        except:
            p["Recovery_Time"] = "0"

    return patients
