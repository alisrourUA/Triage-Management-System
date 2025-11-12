#in this file we compute the "Naive Bayes" to predict the priority level that may be assigned 
# for a patient based on the attributes entered by the user
import math

MIN_SAMPLES_FOR_ML = 10  # min needed samples to start the prediction

#below the functions converts the patient's record into numeric or text features to ensure safety 
def encode_features(p):
    try:
        age = int(float(p.get("Age", p.get("age", 0)) or 0))
    except:
        age = 0
    try:
        hr = int(float(p.get("Heart_Rate", p.get("heart_rate", 0)) or 0))
    except:
        hr = 0
    try:
        bp = int(float(p.get("Blood_Pressure", p.get("blood_pressure", 0)) or 0))
    except:
        bp = 0
    try:
        o2 = int(float(p.get("Oxygen_Level", p.get("oxygen_level", 0)) or 0))
    except:
        o2 = 0

    injury = str(p.get("Injury_Type", p.get("injury", "none"))).strip().lower()
    state = str(p.get("Consciousness", p.get("state", "conscious"))).strip().lower()

    return {
        "age": age,
        "heart": hr,
        "bp": bp,
        "o2": o2,
        "injury": injury,
        "state": state
    }


def nb_train(patients):
    if not patients:
        return None

    class_counts = {}
    like_counts = {}
    features = ["age", "heart", "bp", "o2", "injury", "state"]

    for p in patients:
        try:
            f = encode_features(p)
            c = int(float(p.get("Priority", 0)))
        except Exception:
            continue

        class_counts[c] = class_counts.get(c, 0) + 1

        for fname in features:
            val = f[fname]
            like_counts.setdefault(fname, {})
            like_counts[fname].setdefault(c, {})
            like_counts[fname][c][val] = like_counts[fname][c].get(val, 0) + 1

    return {"class_counts": class_counts, "like_counts": like_counts}

#here's where we predict the priority using the trained model we did above
def nb_predict(model, patient):
    if not model or not patient:
        return None

    f = encode_features(patient)
    features = list(f.keys())
    class_counts = model["class_counts"]
    like_counts = model["like_counts"]

    total_samples = sum(class_counts.values())
    if total_samples == 0:
        return None

    log_probs = {}

    for c in class_counts:
        # Prior probability log(P(class))
        prior = math.log(class_counts[c] / total_samples)
        log_prob = prior

        for fname in features:
            val = f[fname]
            feature_counts = like_counts.get(fname, {}).get(c, {})
            value_count = feature_counts.get(val, 0)
            total_feature_count = sum(feature_counts.values()) or 1

            prob = (value_count + 1) / (total_feature_count + len(feature_counts) + 1)
            log_prob += math.log(prob)

        log_probs[c] = log_prob

    if not log_probs:
        return None

    # then, here according to naive bayes, we pick the class with highest log probability
    predicted_class = max(log_probs, key=log_probs.get)
    return predicted_class



if __name__ == "__main__":
    # minimal test to verify our logic
    dummy_data = [
        {"Age": 60, "Heart_Rate": 120, "Blood_Pressure": 90, "Oxygen_Level": 85, "Injury_Type": "bleeding", "Consciousness": "unconscious", "Priority": 1},
        {"Age": 25, "Heart_Rate": 90, "Blood_Pressure": 120, "Oxygen_Level": 98, "Injury_Type": "minor", "Consciousness": "conscious", "Priority": 3},
        {"Age": 45, "Heart_Rate": 130, "Blood_Pressure": 100, "Oxygen_Level": 92, "Injury_Type": "fracture", "Consciousness": "conscious", "Priority": 2},
    ]

    model = nb_train(dummy_data)
    test_patient = {"Age": 50, "Heart_Rate": 110, "Blood_Pressure": 95, "Oxygen_Level": 90, "Injury_Type": "bleeding", "Consciousness": "unconscious"}
    pred = nb_predict(model, test_patient)
    print("Predicted Priority:", pred)
