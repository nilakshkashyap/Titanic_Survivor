import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
answer = pd.read_csv("gender_submission.csv")

train = train.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis=1)
test = test.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis=1)
X_train = train.drop(["Survived"], axis=1)
Y_train = train["Survived"]
X_test = test
Y_test = answer["Survived"]

X_train["Age"].fillna(value=X_train["Age"].mean(), inplace=True)
X_test["Fare"].fillna(value=X_test["Fare"].mean(), inplace=True)
X_test["Age"].fillna(value=X_test["Age"].mean(), inplace=True)
le = LabelEncoder()
X_train["Sex"] = le.fit_transform(X_train["Sex"])
X_test["Sex"] = le.fit_transform(X_test["Sex"])
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = SVC(kernel="linear", random_state=0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
print("The confusion matrix:-")
print(cm)
print("The accuracy score is " + accuracy_score(Y_test, Y_pred))
