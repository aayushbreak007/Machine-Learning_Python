# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 21:41:10 2018

@author: AAYUSH
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# The digits dataset
digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))

# checking the first 6 images 
'''for index, (image, label) in enumerate(images_and_labels[:6]):
   plt.subplot(2, 3, index + 1)
   plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
   plt.title('Target: %i' % label)'''
   

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)
featureTrain, featureTest, targetTrain, targetTest = train_test_split(data, digits.target, test_size=0.3)
classifier.fit(featureTrain,targetTrain)
predictions = classifier.predict(featureTest)

#diagnal items are the correclty classified items and the off-diagonal items are mis-classified
print(confusion_matrix(targetTest, predictions))
print(accuracy_score(targetTest, predictions))

#testing on few images----98% accuracy

plt.imshow(digits.images[-6], cmap=plt.cm.gray_r, interpolation='nearest')
print("Prediction for test image: ", classifier.predict(data[-6].reshape(1,-1)))

plt.show()
