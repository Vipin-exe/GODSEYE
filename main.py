import os
import json
import joblib

from data.generate_dataset import generate_synthetic_dataset
from preprocessing.pipeline import load_and_preprocess_data
from models.classifiers import get_models
from evaluation.metrics import evaluate_model, plot_model_comparison

def main():
    print("=" * 60)
    print(" Automated Malware Identification ML Pipeline")
    print("=" * 60)
    
    # Setup directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/saved_models', exist_ok=True)
    
    # 1. Dataset Generation
    dataset_path = 'data/malware_dataset.csv'
    if not os.path.exists(dataset_path):
        generate_synthetic_dataset(output_path=dataset_path, n_samples=5000)
    else:
        print(f"[DATA] Dataset already exists at {dataset_path}")
        
    # 2. Preprocessing & Handling Imbalance
    X_train, X_test, y_train, y_test, scaler, selector = load_and_preprocess_data(filepath=dataset_path)
    
    # Save preprocessors for inference
    joblib.dump(scaler, 'outputs/saved_models/scaler.joblib')
    joblib.dump(selector, 'outputs/saved_models/selector.joblib')
    
    # 3. Model Initialization
    models = get_models()
    
    # 4. Training and Evaluation
    print("\n[TRAINING] Starting model training and evaluation...")
    all_results = {}
    
    for model_name, model in models.items():
        print(f"  -> Training {model_name}...")
        
        # Train Model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, model_name=model_name, output_dir='outputs')
        all_results[model_name] = metrics
        
        # Save Model
        model_path = os.path.join('outputs/saved_models', f'{model_name.replace(" ", "_").lower()}.joblib')
        joblib.dump(model, model_path)
        
    # 5. Visualization and Result Aggregation
    plot_model_comparison(all_results, output_dir='outputs')
    
    # Dump metrics to JSON
    results_file = 'outputs/results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\n[PIPELINE COMPLETE] Full pipeline executed successfully.")
    print(f"[INFO] All metrics saved to {results_file}")
    print(f"[INFO] Trained models saved in outputs/saved_models/")
    print(f"[INFO] Plots saved in outputs/")

if __name__ == "__main__":
    main()
