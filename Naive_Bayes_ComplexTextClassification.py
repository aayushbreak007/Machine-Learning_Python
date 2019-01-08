# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:01:32 2018

@author: AAYUSH
"""
from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# we are using "20Newsgroups" datasetfrom sklearn'


#define custom categories
categories=['alt.atheism','soc.religion.christian','comp.graphics','sci.med']

# configure values for the 20-newsgroup dataset
trainingData=fetch_20newsgroups(subset='train',categories=categories,shuffle=True)

#print first 10 lines of the first data
'''
print("\n".join(trainingData.data[1].split("\n")[:30]))
print("Target is:",trainingData.target_names[trainingData.target[1]])'''

# count the word occurances
#tokenizing
countVectorizer=CountVectorizer()
#fitting the data
xTrainCounts=countVectorizer.fit_transform(trainingData.data)


# transform the word occurences (xtrainCounts) into tf-idf
tfidfTransformer=TfidfTransformer() #---don't use TfIdfVectorizer()
xTrainTFIDF=tfidfTransformer.fit_transform(xTrainCounts)

#apply the transformed data to the Naive-bayes model--SUPERVISED
model=MultinomialNB().fit(xTrainTFIDF,trainingData.target)


#TESTING and prediction
testData=['This has nothing to do with church or religion','Software engineering is so good']
xTestCounts=countVectorizer.transform(testData)
xTestTFIDF=tfidfTransformer.transform(xTestCounts)

predicted=model.predict(xTestTFIDF)

for doc, category in zip(testData,predicted):
    print('%r------------>%s'% (doc,trainingData.target_names[category]))

