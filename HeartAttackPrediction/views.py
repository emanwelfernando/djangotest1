from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def prediction(request):
    return render(request, 'prediction.html')
def home(request):
    return render(request, 'home.html')
def result(request):
    data = pd.read_csv(r"C:\Users\Emanwel\Desktop\heart.csv")

    x = data.drop(['oldpeak', 'slp', 'thall', 'caa', 'output'], axis=1)
    y = data['output']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


    model = LogisticRegression()
    model.fit(x_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    val9 = float(request.GET['n9'])

    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8, val9]])


    y_pred=model.predict(x_test)

    result1 = ""
    if pred == [1]:
        result1 = "High Chance of a Heart Attack"
    else:
        result1 = "Low Chance of a Heart Attack"


    return render(request, 'prediction.html', {"result2":result1})
