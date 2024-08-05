from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def train_model(Xtrain, ytrain):
    MLP = MLPClassifier(hidden_layer_sizes=(3), batch_size=50, max_iter=200, random_state=123)  # Increased max_iter to 200
    MLP.fit(Xtrain, ytrain)
    return MLP

def hyperparameter_tuning(X, y):
    MLP = MLPClassifier(random_state=123)
    params = {
        'batch_size': [20, 30, 40, 50],
        'hidden_layer_sizes': [(2,), (3,), (3, 2)],
        'max_iter': [200, 300]  # Increased max_iter options
    }
    grid = GridSearchCV(MLP, params, cv=10, scoring='accuracy')
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_, grid.estimator
