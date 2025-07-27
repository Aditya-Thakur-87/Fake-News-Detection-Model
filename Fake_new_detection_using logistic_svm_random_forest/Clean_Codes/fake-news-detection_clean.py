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


new_data = pd.read_csv(r"C:/Users/thaku/jupyter notebook datasets/Fake news/New_data.csv")


# In[ ]:


new_data.head()


# In[ ]:


new_data.dropna(subset=["clean_text","clean_title"],inplace=True)


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV


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


# In[ ]:


y_predict = random_search.predict(x_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))


# In[ ]:


pipeline2 = Pipeline([("CountVectorizer",CountVectorizer(max_df=1.0,min_df=2,max_features=35000,ngram_range=(1,2))),
                   ("classification",MultinomialNB(alpha=0.09))])


# In[ ]:


pipeline2.fit(x_train,y_train)


# In[ ]:


y_predict = pipeline2.predict(x_test)


# In[ ]:


print(classification_report(y_test, y_predict))


# ## using Text and Title for Vectorization

# In[ ]:


new_data["Content"]=new_data["clean_title"]+" "+new_data["clean_text"]


# In[ ]:


new_data


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

pipeline3 = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=5,max_df=0.75,max_features=None,ngram_range=(1,2))),
    ('clf', MultinomialNB(alpha=0.09))
])


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(new_data["Content"],new_data["label"],test_size=0.2,random_state=33)


# In[ ]:


x_train


# In[ ]:


pipeline3.fit(x_train,y_train)


# In[ ]:


y_predict = pipeline3.predict(x_test)


# In[ ]:


print(classification_report(y_test, y_predict))


# ## Classification Using SVC, Logistic regression and Random Forest Classifier with TFID vectorization
# 

# In[ ]:


tfid_vector =TfidfVectorizer(min_df=5,max_df=0.75,max_features=None,ngram_range=(1,2))


# In[ ]:


x = tfid_vector.fit_transform(new_data["Content"])
x_train,x_test,y_train,y_test = train_test_split(x,new_data["label"],test_size=0.2,random_state=44)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
}
print("starting to train models")
results = []
for name, model in models.items():
    model.fit(x_train, y_train)
    
   
    y_train_pred = model.predict(x_train)
    

    y_test_pred = model.predict(x_test)

    results.append({
        "Model": name,
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Train Precision": precision_score(y_train, y_train_pred, average='weighted'),
        "Test Precision": precision_score(y_test, y_test_pred, average='weighted'),
        "Train Recall": recall_score(y_train, y_train_pred, average='weighted'),
        "Test Recall": recall_score(y_test, y_test_pred, average='weighted'),
        "Train F1 Score": f1_score(y_train, y_train_pred, average='weighted'),
        "Test F1 Score": f1_score(y_test, y_test_pred, average='weighted')
    })
    print("model name: ", name)

results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Train Accuracy", data=results_df)
plt.title("Model Accuracy Comparison (Train Accuracy)")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Test Accuracy", data=results_df)
plt.title("Model Accuracy Comparison (Test Accuracy)")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


rf = RandomForestClassifier(
    class_weight='balanced',
    bootstrap=True,            
    random_state=50,
    n_jobs=4             
)


# In[ ]:


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 6],
    'max_features': ['sqrt', 'log2']
}


# In[ ]:


random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_grid,
    n_iter=50,
    scoring='f1',
    cv=3,
    n_jobs=4,
    verbose=2
)

random_search.fit(x_train, y_train)
best_rf = random_search.best_estimator_


# In[ ]:


y_pred = best_rf.predict(x_test)
y_proba = best_rf.predict_proba(x_test)[:, 1]


# In[ ]:


from sklearn.metrics import classification_report, roc_auc_score
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))


# In[ ]:


print(best_rf)


# In[ ]:




