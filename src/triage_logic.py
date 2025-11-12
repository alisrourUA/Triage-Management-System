# Rule-based triage system
# Priority levels (numeric):
# 1 = Immediate (highest)
# 2 = Urgent
# 3 = Delayed
# 4 = Routine (lowest)

def assign_priority(patient):
   
    def as_int(val, default):
        try:
            return int(float(val))
        except Exception:
            return default

    # extraccting the attributes safely
    hr = as_int(patient.get("Heart_Rate", 0), 0)
    bp = as_int(patient.get("Blood_Pressure", 100), 100)
    o2 = as_int(patient.get("Oxygen_Level", 96), 96)

    state = str(patient.get("Consciousness", "unknown")).strip().lower()
    injury = str(patient.get("Injury_Type", "none")).strip().lower()

    
    # prio 1 → Immediate (critical)
    if state == "unconscious" or o2 < 85 or bp < 80:
        return 1
    # 2 → Urgent
    elif hr > 120 or injury in ("bleeding", "fracture") or (o2 < 90 and bp < 90):
        return 2
    # 3 → Delayed
    elif injury == "minor" or 90 <= hr <= 120:
        return 3
    # 4 → Routine
    else:
        return 4
