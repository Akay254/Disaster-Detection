# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 00:27:56 2020

@author: Akshit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#hhh
import gensim
from gensim.utils import simple_preprocess
import multiprocessing
import re
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
dataset=pd.read_csv('train.csv')



X_train=dataset.iloc[:,[0,3]].values

y_train1=dataset.iloc[0:1000,[4]].values
nlp=[]
for i in range(1000):
    review1 = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review1 = review1.lower()
    review1 = review1.split()
    review1=[lemmatizer.lemmatize(word) for word in review1 if not word in set(stopwords.words('english'))]
    review1= " ".join(review1)
    nlp.append(review1)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(nlp).toarray()

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X,y_train1)
Y1=classifier.predict(X);
import sys
sys.path.append('~/anaconda3/envs/ENV/lib/python3.6/site-packages/hunga_bunga')


"""from sklearn.svm import SVC
classifier2=SVC(kernel='rbf')
classifier2.fit(X,y_train1) """





from sklearn.metrics import confusion_matrix

data=pd.read_csv('test.csv')
X_test=data.iloc[0:1000,3].values

nlpo=[]
for i in range(1000):
    review1 = re.sub('[^a-zA-Z]', ' ', data['text'][i])
    review1 = review1.lower()
    review1 = review1.split()
    review1=[lemmatizer.lemmatize(word) for word in review1 if not word in set(stopwords.words('english'))]
    review1= " ".join(review1)
    nlpo.append(review1)


#before preprocessing of test data
test = cv.fit_transform(nlpo).toarray()
test=test[:,0:4377]
result=classifier.predict(test)



#storing results
daata=pd.read_csv("sample_submission.csv")

for i in range(1000):
    daata['target'][i]=result[i]


    
