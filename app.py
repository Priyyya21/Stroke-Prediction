from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Path to your saved RandomForest model (from your notebook)
MODEL_PATH = "stroke_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


def preprocess_input(form):
    """
    Convert form values to the numeric format expected by the model.
    Feature order (from your notebook):
    ['gender', 'age', 'hypertension', 'heart_disease',
     'ever_married', 'work_type', 'Residence_type',
     'avg_glucose_level', 'bmi', 'smoking_status']
    """
    try:
        gender = int(form.get("gender"))
        age = float(form.get("age"))
        hypertension = int(form.get("hypertension"))
        heart_disease = int(form.get("heart_disease"))
        ever_married = int(form.get("ever_married"))
        work_type = int(form.get("work_type"))
        residence_type = int(form.get("residence_type"))
        avg_glucose_level = float(form.get("avg_glucose_level"))
        bmi = float(form.get("bmi"))
        smoking_status = int(form.get("smoking_status"))

        features = np.array([[gender,
                              age,
                              hypertension,
                              heart_disease,
                              ever_married,
                              work_type,
                              residence_type,
                              avg_glucose_level,
                              bmi,
                              smoking_status]])

        return features

    except Exception as e:
        raise ValueError(f"Invalid input: {e}")


def interpret_risk(probability):
    """
    Turn probability (0–100) into a nice risk label + message.
    """
    if probability < 20:
        return "Low Risk", "Your estimated stroke risk is low. Keep maintaining a healthy lifestyle!"
    elif probability < 50:
        return "Moderate Risk", "There is a moderate risk. Consider regular checkups and healthy habits."
    else:
        return "High Risk", "Your risk appears high. Please consult a medical professional for further evaluation."


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_label = None
    probability = None
    risk_message = None
    error_message = None

    if request.method == "POST":
        if model is None:
            error_message = "Model could not be loaded on the server. Please contact the admin."
        else:
            try:
                features = preprocess_input(request.form)
                # Model from your notebook is a RandomForestClassifier -> supports predict_proba
                proba = model.predict_proba(features)[0][1] * 100  # Probability of stroke (class 1)
                probability = round(proba, 2)
                prediction_label, risk_message = interpret_risk(probability)
            except Exception as e:
                error_message = str(e)

    return render_template(
        "index.html",
        prediction_label=prediction_label,
        probability=probability,
        risk_message=risk_message,
        error_message=error_message
    )


if __name__ == "__main__":
    # For Render: use the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)