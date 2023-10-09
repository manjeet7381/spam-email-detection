#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install scikit-learn numpy pandas


# In[2]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[7]:


df = pd.read_csv('spam.csv',encoding='latin1')


# In[8]:


df.head()


# In[10]:


X = df['v2']  # 'text' is the column containing the email text
y = df['v1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the number of features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[12]:


spam_classifier = MultinomialNB()
spam_classifier.fit(X_train_tfidf, y_train)


# In[13]:


y_pred = spam_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

