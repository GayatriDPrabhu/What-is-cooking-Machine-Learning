
# coding: utf-8

# In[8]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.naive_bayes import GaussianNB

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

# 1. Gaussian NB (Model 1):
# Using test.json file
corpusTrain = traindf['ingredients_string']
corpusTest = testdf['ingredients_string']

vectorizerTrain = TfidfVectorizer(stop_words='english', ngram_range = (1,1), analyzer = "word",
                                   max_df = 0.57, binary = False, token_pattern = r'\w+', sublinear_tf = False)
tfidfTrain = vectorizerTrain.fit_transform(corpusTrain).todense()

vectorizerTest = TfidfVectorizer(stop_words='english')
tfidfTest = vectorizerTrain.transform(corpusTest).todense()

targets_tr = traindf['cuisine']

clf = GaussianNB()
clf.fit(tfidfTrain,targets_tr)

predictions = clf.predict(tfidfTest)

#Print the predictions 
testdf['cuisine'] = predictions

testdf[['id', 'cuisine']].to_csv("gaussianNB1.csv")
# Public Score (gaussianNB1): 0.23481


# In[9]:


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

# 2. Gaussian NB (Model 2):
# Using test.json file
vectorizerTrain = TfidfVectorizer(binary=True)
X = vectorizerTrain.fit_transform(corpusTrain)
X_test = vectorizerTrain.transform(corpusTest)

lblEncoder = LabelEncoder()
train_cuisine = [sample['cuisine'] for sample in json.load(open('train.json'))]
y = lblEncoder.fit_transform(train_cuisine)

clf = GaussianNB()

clf.fit(X.toarray(),y)
y_test = clf.predict(X_test.toarray())
test_cuisine = lblEncoder.inverse_transform(y_test)

test_id = [sample['id'] for sample in json.load(open('test.json'))]

submission_df = pd.DataFrame({'id': test_id, 'cuisine': test_cuisine}, columns=['id', 'cuisine'])
submission_df.to_csv('gaussianNB2.csv', index=False)

# Public Score (gaussianNB2): 0.23310


# In[10]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

import random
import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

# 3. Gaussian NB (Model 3):
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
tfidfTest = vectorizerTrain.transform(corpusTest).toarray()

targets_tr = new_train_df['cuisine']

clf = GaussianNB()
clf.fit(tfidfTrain,targets_tr)

predictions = clf.predict(tfidfTest)

# Print accuracy
Y_test=new_test_df['cuisine']
print(metrics.accuracy_score(Y_test, predictions))

new_test_df['cuisine'] = predictions
new_test_df[['id', 'cuisine']].to_csv("gaussianNB3.csv")
#Accuracy (gaussianNB3): 0.2393854748603352


# In[13]:


# 4. Gaussian NB (Model 4):
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

clf = GaussianNB()
clf = clf.fit(training_data_matrix.toarray(), labels)
print("Training Done")

#get testing data matrix
ids = [item for item in new_test_df['id']]
ingredients = [item for item in new_test_df['ingredients']]
test_data_matrix = scipy.sparse.dok_matrix((len(ingredients), len(unique_ingredients)))

for i, item in enumerate(ingredients):
    for j, ing in enumerate(unique_ingredients):
        if ing in item:
            test_data_matrix[i, j] = 1

res = clf.predict(test_data_matrix.toarray())
print("Predicting Done")

# Print accuracy
Y_test=new_test_df['cuisine']
print(metrics.accuracy_score(Y_test, res))

new_test_df['cuisine'] = res
new_test_df[['id', 'cuisine']].to_csv("gaussianNB4.csv")

print("done")
#Accuracy (gaussianNB4): 0.35726256983240223

