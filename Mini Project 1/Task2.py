import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

f = open("drugs-performance.txt", "w")

dataset = pd.read_csv("drug200.csv")
sex = pd.Categorical(dataset["Sex"].tolist(), categories=['M', 'F'])
bp = pd.Categorical(dataset["BP"].tolist(), ordered=True, categories=['LOW', 'NORMAL', 'HIGH'])
cholesterol = pd.Categorical(dataset["Cholesterol"].tolist(), ordered=True, categories=['NORMAL', 'HIGH'])
drug = pd.Categorical(dataset["Drug"].tolist(), categories=['drugA', 'drugB', 'drugC', 'drugX', 'drugY'])

sex = pd.get_dummies(sex)
bp = pd.get_dummies(bp)
cholesterol = pd.get_dummies(cholesterol)
features = pd.concat([sex,bp,cholesterol], axis=1)
features_train, features_test, drug_train, drug_test = train_test_split(features, drug)


# Gaussian Naive Bayes

gnb = GaussianNB()
gnb.fit(features_train, drug_train)
gnbpredicted = gnb.predict(features_test)

f.write("*******************************\nGaussian Naive Bayes Classifier\n*******************************\n\n")

f.write("Confusion Matrix:\n")
f.write(str(confusion_matrix(drug_test, gnbpredicted)))
f.write("\n\n")

f.write("Analysis:\n")
target= ["drugA","drugB","drugC","drugX","drugY"]
f.write(classification_report(drug_test, gnbpredicted, target_names=target, zero_division=1) + "\n\n")


# Base Decision Tree

bdt = DecisionTreeClassifier()
bdt.fit(features_train, drug_train)
bdtpredicted = bdt.predict(features_test)

f.write("******************\nBase Decision Tree\n******************\n\n")

f.write("Confusion Matrix:\n")
f.write(str(confusion_matrix(drug_test, bdtpredicted)))
f.write("\n\n")

f.write("Analysis:\n")
target= ["drugA","drugB","drugC","drugX","drugY"]
f.write(classification_report(drug_test, bdtpredicted, target_names=target, zero_division=1) + "\n\n")


# Top Decision Tree

parameters = {"max_depth":[40,50], "min_samples_split":[30,40,50], "criterion":["gini","entropy"]}
a = DecisionTreeClassifier()
tdt = GridSearchCV(a, parameters)
tdt.fit(features_train, drug_train)
tdtpredicted = tdt.predict(features_test)

f.write("*****************\nTop Decision Tree\n*****************\n\n")

f.write("Confusion Matrix:\n")
f.write(str(confusion_matrix(drug_test, tdtpredicted)))
f.write("\n\n")

f.write("Analysis:\n")
target= ["drugA","drugB","drugC","drugX","drugY"]
f.write(classification_report(drug_test, tdtpredicted, target_names=target, zero_division=1) + "\n\n")


# Perceptron

per = Perceptron()
per.fit(features_train, drug_train)
perpredicted = per.predict(features_test)

f.write("**********\nPerceptron\n**********\n\n")

f.write("Confusion Matrix:\n")
f.write(str(confusion_matrix(drug_test, perpredicted)))
f.write("\n\n")

f.write("Analysis:\n")
target= ["drugA","drugB","drugC","drugX","drugY"]
f.write(classification_report(drug_test, perpredicted, target_names=target, zero_division=1) + "\n\n")


# Base Multi-Layered Perceptron

mlp = MLPClassifier(hidden_layer_sizes=100, activation="logistic", solver="sgd", max_iter=1000)
mlp.fit(features_train, drug_train)
mlppredicted = mlp.predict(features_test)

f.write("*****************************\nBase Multi-Layered Perceptron\n*****************************\n\n")

f.write("Confusion Matrix:\n")
f.write(str(confusion_matrix(drug_test, mlppredicted)))
f.write("\n\n")

f.write("Analysis:\n")
target= ["drugA","drugB","drugC","drugX","drugY"]
f.write(classification_report(drug_test, mlppredicted, target_names=target, zero_division=1) + "\n\n")


# Top Multi-Layered Perceptron

parameters = {"activation":["identity","logistic","tanh","relu"], "solver":["adam","sgd"], "hidden_layer_sizes":[(30,50),(10,10,10)]}
b = MLPClassifier(max_iter=5000)
tlp = GridSearchCV(b, parameters)
tlp.fit(features_train, drug_train)
tlppredicted = tlp.predict(features_test)

f.write("****************************\nTop Multi-Layered Perceptron\n****************************\n\n")

f.write("Confusion Matrix:\n")
f.write(str(confusion_matrix(drug_test, tlppredicted)))
f.write("\n\n")

f.write("Analysis:\n")
target= ["drugA","drugB","drugC","drugX","drugY"]
f.write(classification_report(drug_test, tlppredicted, target_names=target, zero_division=1) + "\n\n")

f.close()
