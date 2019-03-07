# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 17:11:08 2019

@author: mgelman
"""


#BIG SCORE AND COMPUTATIONAL IMPROVEMENT WHEN CONDITIONING ON DF_MIN 1% SO TAKING OUT SO MANY WORDS THAT ARE TOO SPECIFIC


import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#import packages
import numpy as np
from scipy.sparse import hstack
import operator
import os
import sys
#import pydot
from sklearn.externals.six import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer

#import the fx to scrape data
path = os.path.join(os.path.expanduser("~"),"Documents","GitHub","yelp_dataset") 
sys.path.insert(0, path)
sys.path.insert(0, "C:\Users\mgelman\AppData\Local\Continuum\anaconda2\Library\bin\graphviz")

#STEP 1: LOADING IN DATA - I ALREADY HAVE THE DATA READY TO GO FROM STATA
inputfile=os.path.join(path,"review.json")
#inputfile=os.path.join(path,"review_500k.json")
outputfile=os.path.join(path,"review.pkl")

# load in file
data = pd.read_json(inputfile,lines=True) 


#%% Min and max date?
#only keep date
data=data['date']

data.min(axis=0)
data.max(axis=0)

data['year'].value_counts()

data['year'] = pd.DatetimeIndex(data).year

#%%

#only keep stars and text
data=data[['stars','text']]


#THIS SECTION WHEN LOADING DATA TO PREDICT
#data.to_pickle(outputfile)

#define y as the stars and X as text
y_data=data['stars']
X_data=data['text']
print y_data.value_counts(normalize=True, sort=False)

#create label
star_label = [1,2,3,4,5]

#split into training and test
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=1234)

#vectorize
vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                             stop_words='english',
                             min_df=0.005)
                             #binary=True)
                             #token_pattern=r'\b[^\d\W]+\b') #only words and not numbers
#vectorizer= TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
X_tr = vectorizer.fit_transform(X_train)
X_te = vectorizer.transform(X_test)

vocab = vectorizer.get_feature_names()
vocab_str = [str(x.encode('utf-8')) for x in vocab]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()




#%%

#Use the tree clasifier
#clf = DecisionTreeClassifier(
clf = RandomForestClassifier(
                n_estimators=128,
                n_jobs=-1,
                verbose=1)

#clf = DecisionTreeClassifier(max_leaf_nodes=15)

clf = clf.fit(X_tr, y_train)

 #check which words are the most frequent 
#sum_words = X_tr.sum(axis=0) 
#words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
#words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
# 
# #Look a Top X frequent words
#for word, freq in words_freq[:10]:
#    print(word, freq)  


y_pred = clf.predict(X_te)

#confusion matrix
cm=confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
plot_confusion_matrix(cm, classes=star_label,normalize=True,title='Normalized confusion matrix')



#print cm
#print cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#overall score
training_score = clf.score(X_tr, y_train, sample_weight = None)
testing_score = clf.score(X_te, y_test, sample_weight = None)
print()
print("the training_score is " + str(training_score))
print()
print("the testing_score is " + str(testing_score))


#best features
dictionary = dict(zip(vocab_str,clf.feature_importances_))
sorted_x = sorted(dictionary.items(), key=operator.itemgetter(1),reverse=True)
for x in range(20):
    print sorted_x[x]

#%%
from sklearn.naive_bayes import MultinomialNB    
    
clf = MultinomialNB().fit(X_tr, y_train) #classifying transformed text data to target value 

#confusion matrix
y_pred = clf.predict(X_te)
cm=confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
plot_confusion_matrix(cm, classes=star_label,normalize=True,title='Normalized confusion matrix')


#calculating the mean accuracy on the given test data and labels 
training_score = clf.score(X_tr, y_train, sample_weight = None)
testing_score = clf.score(X_te, y_test, sample_weight = None)
print()
print("the training_score is " + str(training_score))
print()
print("the testing_score is " + str(testing_score))

neg_class_prob_sorted = clf.feature_log_prob_[0, :].argsort()
pos_class_prob_sorted = clf.feature_log_prob_[4, :].argsort()

print(np.take(vocab, neg_class_prob_sorted[-10:]))
print(np.take(vocab, pos_class_prob_sorted[-10:]))


#%%
from sklearn.svm import LinearSVC
    
clf = LinearSVC().fit(X_tr, y_train) #classifying transformed text data to target value 

#confusion matrix
y_pred = clf.predict(X_te)
cm=confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
plot_confusion_matrix(cm, classes=star_label,normalize=True,title='Normalized confusion matrix')


#calculating the mean accuracy on the given test data and labels 
training_score = clf.score(X_tr, y_train, sample_weight = None)
testing_score = clf.score(X_te, y_test, sample_weight = None)
print()
print("the training_score is " + str(training_score))
print()
print("the testing_score is " + str(testing_score))



def plot_coefficients(classifier, feature_names, top_features=10):
 lowstar_coef = classifier.coef_[0,:]*-1
 highstar_coef = classifier.coef_[4,:]
 top_positive_coefficients = np.argsort(highstar_coef)[-top_features:]
 top_negative_coefficients = np.argsort(lowstar_coef)[:top_features]
 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 # create plot
 plt.figure(figsize=(15, 5))
 colors = ['red' if c < 0 else 'blue' for c in lowstar_coef[top_coefficients]]
 plt.bar(np.arange(2 * top_features), lowstar_coef[top_coefficients], color=colors)
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60)
 plt.show()



#plot_coefficients(clf, vocab)
#plot_coefficients(clf, vocab,star_type=4)





#%%
from sklearn.neural_network import MLPClassifier
    
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

clf = clf.fit(X_tr, y_train)

#calculating the mean accuracy on the given test data and labels 
training_score = clf.score(X_tr, y_train, sample_weight = None)
testing_score = clf.score(X_te, y_test, sample_weight = None)
print()
print("the training_score is " + str(training_score))
print()
print("the testing_score is " + str(testing_score))

#%% Compare all the different types 
    
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    KNeighborsClassifier(3),
    LinearSVC(),
    #NuSVC(probability=True),
    GaussianNB(),
    MultinomialNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=8,n_jobs=-1,verbose=1),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    ]
    
# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

denseclass=['GaussianNB','LinearDiscriminantAnalysis','QuadraticDiscriminantAnalysis']

for clf in classifiers:

    name = clf.__class__.__name__
    if name in denseclass:
        X_train=X_tr.toarray()
        X_test=X_te.toarray()
    else:
        X_train=X_tr
        X_test=X_te
        
    clf.fit(X_train, y_train)
    
    
    print("="*30)
    print(name)
    
    print('****Results****')
    acc= clf.score(X_test, y_test, sample_weight = None)
    print("Accuracy: {:.4%}".format(acc))
    
    
    log_entry = pd.DataFrame([[name, acc*100]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)

print log.sort_values(by='Accuracy',ascending=False)






#%%
#Graph the tree
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                  feature_names=vocab_str,
                  filled=True, rounded=True, special_characters=True,
                  # proportion=True
                  )
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graphfile=os.path.join(path,"fig","graph.pdf")
graph[0].write_pdf(graphfile)

    