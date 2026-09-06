import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(y_true, y_pred, model_name, output_dir='outputs'):
    """
    Computes classification metrics and saves a confusion matrix plot for the model.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n[{model_name.upper()}] Evaluation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Generate Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save Plot
    cm_path = os.path.join(output_dir, f'{model_name.replace(" ", "_").lower()}_confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }

def plot_model_comparison(results_dict, output_dir='outputs'):
    """
    Generates a bar chart ranking all models by F1-score.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    models = list(results_dict.keys())
    f1_scores = [results_dict[m]['f1_score'] for m in models]
    
    # Sort models by F1-score for better visualization
    sorted_indices = sorted(range(len(f1_scores)), key=lambda k: f1_scores[k], reverse=True)
    models = [models[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=f1_scores, y=models, hue=models, palette='viridis', legend=False)
    plt.title('Model Comparison by F1-Score')
    plt.xlabel('F1-Score')
    plt.ylabel('Model')
    plt.xlim(0, 1.05)
    
    for i, v in enumerate(f1_scores):
        plt.text(v + 0.01, i, f"{v:.4f}", color='black', va='center')
        
    chart_path = os.path.join(output_dir, 'model_comparison_chart.png')
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    
    print(f"\n[EVALUATION] Saved model comparison chart to {chart_path}")
