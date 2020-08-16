# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 11:43:33 2020

@author: surpraka
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from  sklearn.naive_bayes import MultinomialNB
from  sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
import matplotlib.pyplot as plt

# ------------------------------Data Reading
docs = pd.read_table(r"C:\Users\surpraka\Desktop\MachineLearning\UpGrad\Naive Bayes\Spam_Or_Ham\smsspamcollection\SMSSpamCollection",header=None,names=['Class','sms'])
print(len(docs))

ham_spam = docs.Class.value_counts()
print(ham_spam)

# ------------------------------Data Processing
docs['label'] = docs.Class.map({'ham' : 0 , 'spam' : 1})
docs = docs.drop(columns=['Class'],axis =1)

X = docs.sms
Y = docs.label

X_train,X_test,y_train,y_test = train_test_split(X,Y, random_state = 1,train_size = 0.75,test_size = 0.25)
vec = CountVectorizer(stop_words='english')
vec.fit(X_train)
# print(vec.vocabulary_)
print(len(vec.vocabulary_.keys()))

X_train_transformed = vec.transform(X_train)
X_test_transformed = vec.transform(X_test)

# print(X_train_transformed)

# ------------------------------Multinomial Model Building
mnb = MultinomialNB()
mnb.fit(X_train_transformed,y_train)
prd_class = mnb.predict(X_test_transformed)
prd_prob = mnb.predict_proba(X_test_transformed)


# ------------------------------Multinomial Model Evaluation
acc = metrics.accuracy_score(y_test, prd_class)
print(acc*100)

conf = metrics.confusion_matrix(y_test, prd_class)
print(conf)

TN = conf[0,0]
TP = conf[1,1]
FP = conf[0,1]
FN = conf[1,0]

# True Positive Rate
sensi = TP / float(FN+TP)
print(sensi)

# True Negative Rate
speci = TN / float(TN+FP)
print(speci)

# Actual Positive
precision = TP /float(TP+FP)
print(precision)

#  ROC Curve
false_pos_rate,true_pos_rate,thresholds = metrics.roc_curve(y_test, prd_prob[:,1])
roc_auc = metrics.auc(false_pos_rate, true_pos_rate)
print(roc_auc)

plt.title('ROC')
plt.plot(false_pos_rate,true_pos_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# ------------------------------Bernoullis Model Building
bnb = BernoulliNB()
bnb.fit(X_train_transformed,y_train)
prd_class = bnb.predict(X_test_transformed)
prd_prob = bnb.predict_proba(X_test_transformed)

# ------------------------------Bernoullis Model Evaluation
acc = metrics.accuracy_score(y_test, prd_class)
print(acc*100)

conf = metrics.confusion_matrix(y_test, prd_class)
print(conf)

TN = conf[0,0]
TP = conf[1,1]
FP = conf[0,1]
FN = conf[1,0]

# True Positive Rate
sensi = TP / float(FN+TP)
print(sensi)

# True Negative Rate
speci = TN / float(TN+FP)
print(speci)

# Actual Positive
precision = TP /float(TP+FP)
print(precision)

#  ROC Curve
false_pos_rate,true_pos_rate,thresholds = metrics.roc_curve(y_test, prd_prob[:,1])
roc_auc = metrics.auc(false_pos_rate, true_pos_rate)
print(roc_auc)

plt.title('ROC')
plt.plot(false_pos_rate,true_pos_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
