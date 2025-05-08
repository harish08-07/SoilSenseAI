from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and preprocess dataset
data = pd.read_csv('./data_core.csv')
label_encoders = {}
categorical_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X_crop = data.drop(['Crop Type', 'Fertilizer Name'], axis=1)
y_crop = data['Crop Type']

crop_model = RandomForestClassifier()
crop_model.fit(X_crop, y_crop)

# Feedback function
def get_soil_health_feedback(input_data):
    feedback = []
    if input_data['Nitrogen'] < 10:
        feedback.append("Low nitrogen: apply urea or compost.")
    if input_data['Phosphorous'] < 10:
        feedback.append("Low phosphorous: use DAP or bone meal.")
    if input_data['Potassium'] < 10:
        feedback.append("Low potassium: add potash or wood ash.")
    if input_data['Moisture'] < 30:
        feedback.append("Low moisture: increase irrigation.")
    if input_data['Moisture'] > 70:
        feedback.append("Too wet: improve drainage.")
    if not feedback:
        feedback.append("Soil health is good!")
    return feedback

@app.route("/", methods=["GET", "POST"])
def index():
    crop_result = None
    feedback = []

    if request.method == "POST":
        form = request.form
        user_input = {
            'Temparature': float(form['temperature']),
            'Humidity': float(form['humidity']),
            'Moisture': float(form['moisture']),
            'Soil Type': form['soil_type'],
            'Nitrogen': int(form['nitrogen']),
            'Potassium': int(form['potassium']),
            'Phosphorous': int(form['phosphorous'])
        }

        encoded_input = user_input.copy()
        encoded_input['Soil Type'] = label_encoders['Soil Type'].transform([user_input['Soil Type']])[0]
        df = pd.DataFrame([encoded_input])
        crop_code = crop_model.predict(df)[0]
        crop_result = label_encoders['Crop Type'].inverse_transform([crop_code])[0]

        feedback = get_soil_health_feedback(user_input)

    return render_template("index.html", crop_result=crop_result, feedback=feedback)

if __name__ == "__main__":
    app.run(debug=True)
