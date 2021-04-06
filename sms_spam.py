import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
get_ipython().run_line_magic('matplotlib', 'inline')
sms = pd.read_csv("...../Downloads/project/spam.csv")
sms.head()

sms = sms.drop('2', 1)
sms = sms.drop('3', 1)
sms = sms.drop('4', 1)
sms.head()

sms = sms.rename(columns = {'v1':'label','v2':'message'})
sms.groupby('label').describe()
sms['length'] = sms['message'].apply(len)

sms.head()
sms['length'].plot(bins=100, kind='hist') 
count_Class=pd.value_counts(sms["label"], sort= True)
count_Class.plot(kind = 'bar',color = ["green","red"])
plt.title('Bar Plot')
plt.show();
sms.length.describe()

sms[sms['length'] == 910]['message'].iloc[0]

sms.hist(column='length', by='label', bins=50,figsize=(12,4))

#nlp
import string
from nltk.corpus import stopwords
def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
sms['message'].head(5).apply(text_process)

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(sms['message'])
print(len(bow_transformer.vocabulary_))

sms_bow = bow_transformer.transform(sms['message'])

print('Shape of Sparse Matrix: ', sms_bow.shape)
print('Amount of Non-Zero occurences: ', sms_bow.nnz)
sparsity = (100.0 * sms_bow.nnz / (sms_bow.shape[0] * sms_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(sms_bow)
sms_tfidf = tfidf_transformer.transform(sms_bow)
print(sms_tfidf.shape)

#naive bayes
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(sms_tfidf, sms['label'])

all_predictions = spam_detect_model.predict(sms_tfidf)
print(all_predictions)

from sklearn.metrics import classification_report
print (classification_report(sms['label'], all_predictions))

from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(sms['message'], sms['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

# DATA PIPELINE

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(predictions,label_test))




