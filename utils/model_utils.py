from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def scale_data(xtrain, xtest):
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    Xtrain = scaler.transform(xtrain)
    Xtest = scaler.transform(xtest)
    return Xtrain, Xtest

def evaluate_model(ytest, ypred):
    cm = confusion_matrix(ytest, ypred)
    accuracy = accuracy_score(ytest, ypred)
    return cm, accuracy

def plot_loss_curve(loss_values):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
