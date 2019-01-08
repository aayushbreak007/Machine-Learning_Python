# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:41:37 2018

@author: AAYUSH
"""

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer=TfidfVectorizer()

tfidf=vectorizer.fit_transform(["I like machine learning and clustering algorithms",
                                "Apples, oranges and any kind of fruits are healthy",
                                "Is it feasible with machine learning algorithm?",
                                "My family is happy because of the healthy fruits"])

# THIS WILL RETURN THE DOCUMENT TERM MATRIX
# A DOC TERM MATRIX CONTAINS FREQUENCY OF WORDS
#THEN IT WILL CALUCLTAE THE TF-IDF VALUES

#similarity matrix
# note---here T is the tranpose of A matrix
print((tfidf*tfidf.T).A)