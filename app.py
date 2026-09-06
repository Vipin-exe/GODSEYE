import os
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from utils.extractor import extract_features_from_file

app = Flask(__name__)
app.secret_key = 'super_secret_ml_key'

# Setup upload folder
basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load ML Models and Preprocessors
# We use Random Forest as the default classifier for this demo
print("[APP] Loading ML Models...")
try:
    scaler = joblib.load('outputs/saved_models/scaler.joblib')
    selector = joblib.load('outputs/saved_models/selector.joblib')
    model = joblib.load('outputs/saved_models/random_forest.joblib')
    print("[APP] Models loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}. Did you run main.py first?")
    scaler, selector, model = None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
        
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
        
    if file and model and scaler and selector:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 1. Extract Features (Simulated)
        feature_dict, feature_list = extract_features_from_file(filepath)
        
        # 2. Preprocess (Scale and Select)
        X_raw = np.array([feature_list])
        X_scaled = scaler.transform(X_raw)
        X_selected = selector.transform(X_scaled)
        
        # 3. Predict
        prediction = model.predict(X_selected)[0]
        probabilities = model.predict_proba(X_selected)[0]
        
        # Class 0: Benign, Class 1: Malware
        result_label = "Malware" if prediction == 1 else "Benign"
        confidence = probabilities[prediction] * 100
        
        # Cleanup uploaded file
        os.remove(filepath)
        
        return render_template('result.html', 
                               filename=filename,
                               result=result_label,
                               confidence=f"{confidence:.2f}%",
                               features=feature_dict)
                               
    flash('Model not loaded properly. Please run the ML pipeline first.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
