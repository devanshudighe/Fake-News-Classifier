"""
Created on Mon Dec 16 18:26:24 2019

@author: devanshu
"""
# Detection of News - Real/Fake 

#Importing libraries
import numpy as np
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("news.csv")
dataset.head()
dataset.shape

#Get label
labels = dataset.label
labels.head()

#Split the dataset into Train,Test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset['text'], labels, test_size = 0.2, random_state = 7)

#Initialise the TF-IDFVectoriser
from sklearn.feature_extraction.text import TfidfVectorizer
"""
    The TfidfVectorizer() takes 2 arguments 'stop_words' and 'max_df'
    stop_words: common words that are needed to be filtered out (eg: a,an,the)
    max_df: It sets the threshold for the vocabulary to ignore the document
            frequency above the given value.
"""
tdidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)

#Fit the dataset to this vectorizer
X_train_tdidf = tdidf_vectorizer.fit_transform(X_train)
X_test_tdidf = tdidf_vectorizer.transform(X_test)

#Initialize the PassiveAgressiveClassifier
"""
    PassiveAggressiveClassifier uses 1 parameter - 
    max_iter : It represents the number of passes/iterations over the train_set
    
"""
from sklearn.linear_model import PassiveAggressiveClassifier
pass_aggr_classifier = PassiveAggressiveClassifier(max_iter = 50)
pass_aggr_classifier.fit(X_train_tdidf,y_train)

#Predict on the test set
y_pred = pass_aggr_classifier.predict(X_test_tdidf)

#confusion matrix
"""
    True Positives:593
    True Negatives:586
    False Positives:45
    False Negatives:43
"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

"""
    We found out the accuracy to be 93%
    
"""
