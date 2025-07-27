#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv("C:/Users/thaku/jupyter notebook datasets/WELFake_Dataset.csv")


# In[ ]:


data.sample(n=10)


# In[ ]:


data.info()


# In[ ]:


data.drop("Unnamed: 0",axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.dropna(axis=0,inplace=True)


# In[ ]:


data.duplicated().sum()


# In[ ]:


data.drop_duplicates(inplace=True)


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


data['label'].value_counts().values


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


ax= sns.barplot(x=data['label'].value_counts().index,y=data['label'].value_counts().values)
plt.title("Target Class Distribution")
plt.xlabel("Target Classes")
plt.ylabel("Count")
plt.xticks(ticks=[0,1],labels=['Fake','Real'])
plt.tight_layout()
for i, value in enumerate(data['label'].value_counts().values):
    ax.text(i, value + 2, str(value), ha='center', va='bottom', fontsize=10)


# In[ ]:


import string
exclude =string.punctuation


# In[ ]:


exclude


# In[ ]:


def remove_punctuation(text):
    text=text.translate(str.maketrans('', '', exclude))
    return text.lower().strip()


# In[ ]:


data['text']=data['text'].apply(remove_punctuation)
data['title']=data['title'].apply(remove_punctuation)


# In[ ]:


data.head(5)


# In[ ]:


from bs4 import BeautifulSoup

def has_html_tags_bs(text):
    return text != BeautifulSoup(text, "html.parser").get_text()


# In[ ]:


check_html_text= data['text'].apply(has_html_tags_bs)
check_html_title = data['title'].apply(has_html_tags_bs)
print(check_html_text.value_counts())
print(check_html_title.value_counts())


# In[ ]:


pip install spacy


# In[ ]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[ ]:


import spacy
import os
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

def batch_lemmatize(texts, batch_size=800, n_process=4):
    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        lemmas = [
            token.lemma_.lower() for token in doc
            if token.is_alpha and not token.is_stop and len(token) > 2
        ]
        results.append(" ".join(lemmas))
    return results


# In[ ]:


texts = data["text"].astype(str).tolist()
titles = data["title"].astype(str).tolist()


# In[ ]:


data['clean_title'] = batch_lemmatize(titles)
data['clean_text'] = batch_lemmatize(texts)


# In[ ]:


pd.set_option("display.max_colwidth",None)
data.head(5)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


data.to_csv("New_data.csv",index=False)


# In[ ]:


data.sample(4)


# ## Using Text only for Vectorization

# In[ ]:


import pandas as pd
new_data = pd.read_csv(r"/New_data.csv")


# In[ ]:


new_data.head()


# In[ ]:


new_data.dropna(subset=["clean_text","clean_title"],inplace=True)


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


model_NB= Pipeline([("CountVectorizer",CountVectorizer()),
                   ("classification",MultinomialNB())])


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(new_data["clean_text"],new_data["label"],test_size=0.2,random_state=1)


# In[ ]:


x_train


# In[ ]:


param_grid = {
    'CountVectorizer__max_df': [0.8, 1.0],
    'CountVectorizer__min_df': [2, 5],
    'CountVectorizer__ngram_range': [(1, 1), (1, 2)],
    'CountVectorizer__max_features': [10000, 20000],
    'classification__alpha': [0.1, 0.5, 1.0]
}


# In[ ]:


random_search = RandomizedSearchCV(
    model_NB,
    param_distributions=param_grid,
    n_iter=24,
    scoring='accuracy',
    cv=3,
    verbose=2,
    n_jobs=4,
    random_state=1
)
random_search.fit(x_train, y_train)
print("Best Params:", random_search.best_params_)


# ## Using TFID with naive bayes classifier

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.75, min_df=5, ngram_range=(1, 2))),
    ('clf', MultinomialNB(alpha=np.float64(0.7000000000000001)))
])


# In[ ]:


import numpy as np
param_dist = {
    'tfidf__max_df': [0.75, 0.9, 1.0],
    'tfidf__min_df': [1, 2, 5],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__max_features': [10000, 20000, None],
    'clf__alpha': np.linspace(0.1, 1.0, 10)  # Smoothing parameter
}


# In[ ]:


random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=1
)


# In[ ]:


y_predict =pipeline.predict(x_test)


# In[ ]:


pipeline.fit(x_train,y_train)


# In[ ]:


random_search.best_params_


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,y_predict))


# In[ ]:


pip install gensim scikit-learn nltk


# In[ ]:


import spacy
from gensim.models import Word2Vec


# In[ ]:


new_data["clean_text"][0]


# In[ ]:


nlp = spacy.load("en_core_web_sm")
def spacy_tokenizer(text):
  doc = nlp(text.lower())
  return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]


# In[ ]:


tokenized_content = new_data["clean_text"].apply(spacy_tokenizer).tolist()


# In[ ]:


from gensim.models import Word2Vec

w2v_model = Word2Vec(
    sentences=tokenized_content,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=0
)


# In[ ]:


import numpy as np
def get_avg_w2v(tokens, model):
    vec = np.zeros(model.vector_size)
    count = 0
    for word in tokens:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    return vec / count if count > 0 else vec

X = np.array([get_avg_w2v(doc, w2v_model) for doc in tokenized_content])


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,new_data["label"],test_size=0.2,random_state=11)


# In[ ]:


import pickle
with open("tokenized_content.pkl",'wb') as file:
  pickle.dump(tokenized_content,file)


# In[ ]:


x_train.shape


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# # Random Forest Classifier with Word2Vec

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# Define model
rf = RandomForestClassifier(random_state=78)

# Hyperparameter grid
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 8],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False]
}

# Randomized Search
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=80,
    cv=3,
    verbose=2,
    n_jobs=-1,
    scoring='f1'
)

# Fit model
random_search.fit(x_train, y_train)

# Best model
best_rf = random_search.best_estimator_

print(best_rf)


# In[ ]:


y_pred = best_rf.predict(x_test)
y_proba = best_rf.predict_proba(x_test)[:, 1]


# In[ ]:


from sklearn.metrics import classification_report, roc_auc_score
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))


# In[ ]:




