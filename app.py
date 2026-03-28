import os
import pickle
import warnings
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Load models once at startup ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

with open(os.path.join(MODEL_DIR, "Scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    model_knn5 = pickle.load(f)   # KNN n_neighbors=5

with open(os.path.join(MODEL_DIR, "model1.pkl"), "rb") as f:
    model_knn7 = pickle.load(f)   # KNN n_neighbors=7

NUM_COLS = list(scaler.feature_names_in_)          # ['pclass','age','sibsp','parch','fare']
CAT_COLS = list(encoder.feature_names_in_)         # ['sex','embarked','class','who','adult_male','alive','alone']

# ── Helper ────────────────────────────────────────────────────────────────────
def build_features(data: dict) -> np.ndarray:
    """Convert raw form data → numpy feature array ready for prediction."""
    row = {
        "pclass":     int(data["pclass"]),
        "age":        float(data["age"]),
        "sibsp":      int(data["sibsp"]),
        "parch":      int(data["parch"]),
        "fare":       float(data["fare"]),
        "sex":        data["sex"],                              # 'female' | 'male'
        "embarked":   data["embarked"],                        # 'C' | 'Q' | 'S'
        "class":      data["class"],                           # 'First' | 'Second' | 'Third'
        "who":        data["who"],                             # 'child' | 'man' | 'woman'
        "adult_male": data["adult_male"].lower() == "true",   # bool
        "alive":      data["alive"],                           # 'yes' | 'no'
        "alone":      data["alone"].lower() == "true",         # bool
    }

    df = pd.DataFrame([row])
    scaled  = scaler.transform(df[NUM_COLS])
    encoded = encoder.transform(df[CAT_COLS])
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    return np.hstack([scaled, encoded])


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        X = build_features(data)

        model_choice = data.get("model", "knn5")
        model = model_knn5 if model_choice == "knn5" else model_knn7

        prediction = int(model.predict(X)[0])
        proba      = model.predict_proba(X)[0]   # [prob_perished, prob_survived]

        return jsonify({
            "survived":          prediction == 1,
            "survival_prob":     round(float(proba[1]), 4),
            "perished_prob":     round(float(proba[0]), 4),
            "model_used":        f"KNN (k={model.n_neighbors})",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)