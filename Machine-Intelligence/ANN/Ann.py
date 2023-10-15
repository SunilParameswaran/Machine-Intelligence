import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,confusion_matrix
import torch

# Split the data into training and testing sets
# input: 1) x: list/ndarray (features)
#        2) y: list/ndarray (target)
# output: split: tuple of X_train, X_test, y_train, y_test
def split_and_standardize(X,y):
    #TODO
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform (X_test)
    #y_train = scaler.fit_transform(y_train)
    #y_test = scaler.fit_transform(y_test)
    return X_train,X_test, y_train, y_test
    pass





# Create and train 2 MLP classifier(of 3 hidden layers each) with different parameters
# input:  1) X_train: list/ndarray
#         2) y_train: list/ndarray

# output: 1) models: model1,model2 - tuple
def create_model(X_train,y_train):
    #TODO
    model1 = MLPClassifier(hidden_layer_sizes=(3,3,3),activation='relu',max_iter=1000,random_state=42)
    model2 = MLPClassifier(hidden_layer_sizes=(3,3,3),activation='logistic',max_iter=1000,random_state=42)
    model1.fit(X_train,y_train)
    model2.fit(X_train,y_train)

    return model1,model2
    pass




# create model with parameters
# input  : 1) model: MLPClassifier after training
#          2) X_train: list/ndarray
#          3) y_train: list/ndarray
# output : 1) metrics: tuple - accuracy,precision,recall,fscore,confusion matrix
def predict_and_evaluate(model,X_test,y_test):
    #TODO
    y_pred_model1 = model.predict(X_test)
    accuracy_model1 = accuracy_score(y_test, y_pred_model1)
    precision_model1 = precision_score(y_test, y_pred_model1,average='weighted')
    recall_model1 = recall_score(y_test, y_pred_model1,average='weighted')
    f1_score_model1 = f1_score(y_test, y_pred_model1,average='weighted')
    confusion_matrix_model1 = confusion_matrix(y_test, y_pred_model1)

    # print("Model 1:")
    #print(f"Accuracy: {accuracy_model1}")
    # print(f"Precision: {precision_model1}")
    # print(f"Recall: {recall_model1}")
    # print(f"F1 Score: {f1_score_model1}")
    # print("Confusion Matrix:")
    # print(confusion_matrix_model1)
    return accuracy_model1,precision_model1,recall_model1,f1_score_model1,confusion_matrix_model1

    pass




        

