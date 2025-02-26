from flask import Flask, render_template, request
import pickle
import numpy as np
import webbrowser
import threading

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('cvd_risk_model.pkl', 'rb'))

# Automatically open browser when server starts
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input from form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])

        # Prepare input for model
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang]])  
        prediction = model.predict(input_data)

        # Format prediction output
        result = "Yes, risk detected! 🚨" if prediction[0] == 1 else "No, you are safe! ✅"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()  # Delay to ensure server starts first
    app.run(debug=True)