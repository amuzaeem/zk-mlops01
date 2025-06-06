from flask import Flask, request, jsonify, render_template_string
from flask import render_template
import joblib
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model and metrics
try:
    model = joblib.load('model.pkl')
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None  # Handle the case where the model cannot be loaded
with open('metrics.pkl', 'rb') as file:
    metrics = pickle.load(file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Create a DataFrame from the input data
    input_data = pd.DataFrame([data])

    # Ensure correct column order
    input_data = input_data[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
                             'Alkaline_Phosphatase',
                             'Alanine_Aminotransferase',
                             'Aspartate_Aminotransferase',
                             'Total_Proteins',
                             'Albumin', 'Albumin_and_Globulin_Ratio']]

    # Make prediction
    prediction = model.predict(input_data)

    result = (
        "has liver disease"
        if prediction[0] == 1
        else "does not have liver disease"
    )

    # Detailed message
    message = {
        "prediction": result,
        "details": f"The model predicts that the patient {result}.\
        This is based on the input values provided."
    }

    # Send the prediction result back
    return jsonify(message)


@app.route('/metrics')
def metrics_route():
    metrics_html = """
    <h1>Model Performance Metrics</h1>
    <table border="1">
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        {% for metric, value in metrics.items() %}
        <tr>
            <td>{{ metric }}</td>
            <td>{{ value }}</td>
        </tr>
        {% endfor %}
    </table>
    <br>
    <button onclick="window.location.href='/'">Back</button>
    """
    return render_template_string(metrics_html, metrics=metrics)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
