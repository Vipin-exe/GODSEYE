import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(filepath='data/malware_dataset.csv', test_size=0.2, random_state=42):
    """
    Loads dataset, splits into train/test, scales features, selects top features,
    and handles class imbalance using SMOTE.
    """
    print(f"[PREPROCESSING] Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    # 1. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"[PREPROCESSING] Initial training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # 2. Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Feature Selection (SelectKBest)
    # Selecting the top 8 features for demonstration
    k = min(8, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"[PREPROCESSING] Selected {k} features: {selected_features}")
    
    # 4. Handling Class Imbalance (SMOTE)
    # The report explicitly calls out class imbalance as a challenge.
    print(f"[PREPROCESSING] Class distribution before SMOTE: Benign={sum(y_train==0)}, Malware={sum(y_train==1)}")
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
    print(f"[PREPROCESSING] Class distribution after SMOTE: Benign={sum(y_train_resampled==0)}, Malware={sum(y_train_resampled==1)}")
    
    return X_train_resampled, X_test_selected, y_train_resampled, y_test, scaler, selector
