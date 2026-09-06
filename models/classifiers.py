from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_models(random_state=42):
    """
    Returns a dictionary of the 5 requested machine learning models initialized
    with their respective hyperparameters.
    """
    models = {
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel='rbf', probability=True, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state)
    }
    return models
