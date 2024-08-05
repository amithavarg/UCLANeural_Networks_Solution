import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.data_utils import load_data, preprocess_data, split_data
from utils.model_utils import scale_data, evaluate_model, plot_loss_curve
from models.mlp_model import train_model, hyperparameter_tuning
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main():
    # Suppress convergence warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

    # Load Data
    df = load_data('data/Admission.csv')

    # Preprocess Data
    df = preprocess_data(df)

    # Initial Data Check
    print("Initial Data Check:")
    print(df.head())
    print(df.describe().T)
    print(df.shape)
    df.info()

    # Visualize Pairplot
    plt.figure(figsize=(15, 8))
    sns.scatterplot(data=df, x='GRE_Score', y='TOEFL_Score', hue='Admit_Chance')
    plt.show()

    # Split Data
    x, y = split_data(df)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=123)

    # Scale Data
    Xtrain, Xtest = scale_data(xtrain, xtest)

    # Train Model
    MLP = train_model(Xtrain, ytrain)

    # Make Predictions
    ypred = MLP.predict(Xtest)

    # Evaluate Model
    cm, accuracy = evaluate_model(ytest, ypred)
    print("Confusion Matrix:\n", cm)
    print("Accuracy:", accuracy)

    # Plot Loss Curve
    plot_loss_curve(MLP.loss_curve_)

    # Hyperparameter Tuning
    best_params, best_score, best_estimator = hyperparameter_tuning(x, y)
    print("Best Params:", best_params)
    print("Best Score:", best_score)
    print("Best Estimator:", best_estimator)

if __name__ == "__main__":
    main()
