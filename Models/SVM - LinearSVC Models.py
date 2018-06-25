
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn import svm

traindf = pd.read_json("train.json")
traindf['ingredients_clean_string'] = [' , '.join(ingredients).strip() for ingredients in traindf['ingredients']]
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line))
                                 for line in ingredients]).strip()
                                 for ingredients in traindf['ingredients']]

testdf = pd.read_json("test.json")
testdf['ingredients_clean_string'] = [' , '.join(ingredients).strip() for ingredients in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line))
                                for line in ingredients]).strip()
                                for ingredients in testdf['ingredients']]

# 1. SVM (Model 1):
# Using test.json file
corpusTrain = traindf['ingredients_string']
corpusTest = testdf['ingredients_string']

vectorizerTrain = TfidfVectorizer(stop_words='english', ngram_range = (1,1), analyzer = "word",
                                   max_df = 0.57, binary = False, token_pattern = r'\w+', sublinear_tf = False)
tfidfTrain = vectorizerTrain.fit_transform(corpusTrain).todense()

vectorizerTest = TfidfVectorizer(stop_words='english')
tfidfTest = vectorizerTrain.transform(corpusTest)

targets_tr = traindf['cuisine']

clf = svm.LinearSVC()
#clf = svm.LinearSVC(random_state=1, C=0.4, penalty="l2", dual=False)
parameters = {'C':[1, 10]}
classifier = grid_search.GridSearchCV(clf, parameters)
classifier = classifier.fit(tfidfTrain,targets_tr)

predictions = classifier.predict(tfidfTest)

#Print the predictions 
testdf['cuisine'] = predictions

testdf[['id', 'cuisine']].to_csv("svm-linearSVC1_new.csv")
# Public Score (svm-linearSVC1): 0.78901
# Public Score (svm-linearSVC1_new): 0.78891


# In[3]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

import json
import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

traindf = pd.read_json("train.json")

traindf['ingredients_clean_string'] = [' , '.join(ingredients).strip() for ingredients in traindf['ingredients']]
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line))
                                 for line in ingredients]).strip()
                                 for ingredients in traindf['ingredients']]

testdf = pd.read_json("test.json")
testdf['ingredients_clean_string'] = [' , '.join(ingredients).strip() for ingredients in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line))
                                for line in ingredients]).strip()
                                for ingredients in testdf['ingredients']]

# 2. SVM (Model 2):
# Using test.json file
vectorizerTrain = TfidfVectorizer(binary=True)
X = vectorizerTrain.fit_transform(corpusTrain)
X_test = vectorizerTrain.transform(corpusTest)

lblEncoder = LabelEncoder()
train_cuisine = [sample['cuisine'] for sample in json.load(open('train.json'))]
y = lblEncoder.fit_transform(train_cuisine)

#clf = svm.LinearSVC()
clf = svm.LinearSVC(random_state=1, C=0.4, penalty="l2", dual=False)

clf.fit(X.toarray(),y)
y_test = clf.predict(X_test.toarray())
test_cuisine = lblEncoder.inverse_transform(y_test)

test_id = [sample['id'] for sample in json.load(open('test.json'))]

submission_df = pd.DataFrame({'id': test_id, 'cuisine': test_cuisine}, columns=['id', 'cuisine'])
submission_df.to_csv('svm-linearSVC2_new.csv', index=False)

# Public Score (svm-linearSVC2): 0.78690
# Public Score (svm-linearSVC2_new): 0.78730


# In[5]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

import random
import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn import metrics
from sklearn import svm

# 3. SVM (Model 3):
# Using random 30% training data (from train.json) for testing 
# and the remaining 70% training data (from train.json) for training

traindf = pd.read_json("train.json")

new_test_df=pd.DataFrame()
new_train_df=pd.DataFrame()

testRows = random.sample(list(traindf.index), round(0.3*11932))
new_test_df = traindf.loc[testRows]
new_train_df = traindf.drop(testRows)

new_train_df['ingredients_clean_string'] = [' , '.join(ingredients).strip() for ingredients in new_train_df['ingredients']]
new_train_df['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line))
                                 for line in ingredients]).strip()
                                 for ingredients in new_train_df['ingredients']]

new_test_df['ingredients_clean_string'] = [' , '.join(ingredients).strip() for ingredients in new_test_df['ingredients']]
new_test_df['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line))
                                for line in ingredients]).strip()
                                for ingredients in new_test_df['ingredients']]

corpusTrain = new_train_df['ingredients_string']
corpusTest = new_test_df['ingredients_string']

vectorizerTrain = TfidfVectorizer(stop_words='english', ngram_range = (1,1), analyzer = "word",
                                   max_df = 0.57, binary = False, token_pattern = r'\w+', sublinear_tf = False)
tfidfTrain = vectorizerTrain.fit_transform(corpusTrain).todense()

vectorizerTest = TfidfVectorizer(stop_words='english')
tfidfTest = vectorizerTrain.transform(corpusTest)

targets_tr = new_train_df['cuisine']

#clf = svm.LinearSVC()
clf = svm.LinearSVC(random_state=1, C=0.4, penalty="l2", dual=False)

parameters = {'C':[1, 10]}
classifier = grid_search.GridSearchCV(clf, parameters)
classifier = classifier.fit(tfidfTrain,targets_tr)

predictions = classifier.predict(tfidfTest)

# Print accuracy
Y_test=new_test_df['cuisine']
print(metrics.accuracy_score(Y_test, predictions))

new_test_df['cuisine'] = predictions
new_test_df[['id', 'cuisine']].to_csv("svm-linearSVC3_new.csv")
#Accuracy (svm-linearSVC3): 0.7854748603351955
#Accuracy (svm-linearSVC3_new): 0.7935754189944134


# In[10]:


# 4. SVM (Model 4):
# Using random 30% training data (from train.json) for testing 
# and the remaining 70% training data (from train.json) for training

#Using sparse matrix

get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

from sklearn import svm
import scipy.sparse
import csv
import pandas as pd
import random
from sklearn import metrics

#My implementation:
traindf = pd.read_json("train.json")

new_test_df=pd.DataFrame()
new_train_df=pd.DataFrame()

#testing: random 30%, training: remaining 70%
testRows = random.sample(list(traindf.index), round(0.3*11932))
new_test_df = traindf.loc[testRows]
new_train_df = traindf.drop(testRows)

#get training data matrix
labels = [item for item in new_train_df['cuisine']]
ingredients = [item for item in new_train_df['ingredients']]
unique_ingredients = set(inner_item for outer_item in ingredients for inner_item in outer_item)
training_data_matrix = scipy.sparse.dok_matrix((len(ingredients), len(unique_ingredients)))

for i, item in enumerate(ingredients):
    for j, ing in enumerate(unique_ingredients):
        if ing in item:
            training_data_matrix[i, j] = 1

clf = svm.LinearSVC(random_state=1, C=0.4, penalty="l2", dual=False)
#clf = svm.LinearSVC()
clf = clf.fit(training_data_matrix, labels)
print("Training Done")

#get testing data matrix
ids = [item for item in new_test_df['id']]
ingredients = [item for item in new_test_df['ingredients']]
test_data_matrix = scipy.sparse.dok_matrix((len(ingredients), len(unique_ingredients)))

for i, item in enumerate(ingredients):
    for j, ing in enumerate(unique_ingredients):
        if ing in item:
            test_data_matrix[i, j] = 1

res = clf.predict(test_data_matrix)
print("Predicting Done")

# Print accuracy
Y_test=new_test_df['cuisine']
print(metrics.accuracy_score(Y_test, res))

new_test_df['cuisine'] = res
new_test_df[['id', 'cuisine']].to_csv("svm-linearSVC4_new.csv")

print("done")
#Accuracy (svm-linearSVC4): 0.7659217877094973
#Accuracy (svm-linearSVC4_new): 0.779608938547486

