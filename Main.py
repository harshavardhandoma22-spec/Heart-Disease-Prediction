
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import random

main = tkinter.Tk()
main.title("Machine Learning for Real-Time Heart Disease Prediction") #designing main screen
main.geometry("1300x1200")

global dataset
global filename
global X, Y
global le
global X_train, X_test, y_train, y_test, labels, scaler, xg_cls
global accuracy, precision, recall, fscore

def upload():
    global filename, dataset, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))
    labels = np.unique(dataset['Label'])
    text.insert(END,"\n\nLabels : "+str(labels))

    #plotting graph with various diseases and its count found in data
    attack = dataset.groupby('Label').size()
    attack.plot(kind="bar")
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Dataset Labels Found in Dataset')
    plt.show()

def preprocessing():
    global dataset, X, Y, le

    text.delete('1.0', END)

    le = LabelEncoder()
    Y = le.fit_transform(dataset['Label'].astype(str))

    X = dataset.drop(columns=['Label']).fillna(0).to_numpy()

    text.insert(
        END,
        "Dataset preprocessing completed.\n"
        "Features and labels prepared successfully.\n"
    )
    text.insert(END, f"Number of records: {X.shape[0]}\n")
    text.insert(END, f"Number of features: {X.shape[1]}\n")
    text.insert(END, f"Classes: {list(le.classes_)}\n")

def trainTestSplit():
    global X, Y
    global X_train, X_test, y_train, y_test
    global scaler

    text.delete('1.0', END)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    text.insert(
        END,
        "Dataset split completed.\n"
        "Training data: 80%\n"
        "Validation data: 20%\n\n"
    )
    text.insert(END, f"Training records: {X_train.shape[0]}\n")
    text.insert(END, f"Validation records: {X_test.shape[0]}\n")

def runXGboost():
    global X_train, X_test, y_train, y_test
    global xg_cls, accuracy, precision, recall, fscore

    text.delete('1.0', END)

    xg_cls = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
    )

    xg_cls.fit(X_train, y_train)
    predictions = xg_cls.predict(X_test)

    accuracy = accuracy_score(y_test, predictions) * 100
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    ) * 100
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    ) * 100
    fscore = f1_score(
        y_test, predictions, average="macro", zero_division=0
    ) * 100

    text.insert(END, f"XGBoost Accuracy: {accuracy:.2f}%\n")
    text.insert(END, f"XGBoost Precision: {precision:.2f}%\n")
    text.insert(END, f"XGBoost Recall: {recall:.2f}%\n")
    text.insert(END, f"XGBoost Macro F1-score: {fscore:.2f}%\n")

    class_ids = np.arange(len(le.classes_))
    conf_matrix = confusion_matrix(
        y_test, predictions, labels=class_ids
    )

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        conf_matrix,
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        annot=True,
        cmap="viridis",
        fmt="g"
    )
    plt.title("XGBoost Confusion Matrix")
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.tight_layout()
    plt.show()

def graph():
    global accuracy, precision, recall, fscore
    height = [accuracy, precision, recall, fscore]
    bars = ('Accuracy', 'Precision', 'Recall', 'Fscore')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Comparison Metrics")
    plt.ylabel("Values")
    plt.title("XGBoost Accuracy, Precision, Recall and FSCORE Comparison Graph")
    plt.show()



def predict():
    global xg_cls, scaler, le

    if "xg_cls" not in globals() or "scaler" not in globals():
        messagebox.showerror(
            "Model not ready",
            "Please preprocess the dataset, split it, and train the model first."
        )
        return

    filename = filedialog.askopenfilename(
        initialdir="Dataset",
        title="Select test CSV",
        filetypes=[("CSV files", "*.csv")]
    )

    if not filename:
        return

    try:
        test_df = pd.read_csv(filename)

        if "Label" in test_df.columns:
            test_df = test_df.drop(columns=["Label"])

        expected_columns = dataset.drop(columns=["Label"]).columns
        test_df = test_df.reindex(columns=expected_columns).fillna(0)

        test_scaled = scaler.transform(test_df)
        predictions = xg_cls.predict(test_scaled)

        text.delete("1.0", END)

        for i, class_id in enumerate(predictions):
            predicted_label = le.inverse_transform([int(class_id)])[0]
            text.insert(
                END,
                f"Test record {i + 1} ==> Predicted class: {predicted_label}\n\n"
            )

    except Exception as error:
        messagebox.showerror("Prediction error", str(error))

    
    

font = ('times', 13, 'bold')
title = Label(main, text='Machine Learning for Real-Time Heart Disease Prediction')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=24,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload ECG Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

preButton = Button(main, text="Dataset Preprocessing", command=preprocessing)
preButton.place(x=300,y=100)
preButton.config(font=font1) 

trainButton = Button(main, text="Train & Test Split", command=trainTestSplit)
trainButton.place(x=490,y=100)
trainButton.config(font=font1) 

xgboostButton = Button(main, text="Run XGBoost Algorithm", command=runXGboost)
xgboostButton.place(x=680,y=100)
xgboostButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=150)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Heart Disease from Test Data", command=predict)
predictButton.place(x=300,y=150)
predictButton.config(font=font1)


#main.config(bg='OliveDrab2')
main.mainloop()
