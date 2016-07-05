import numpy as np
import pandas as pd
from sqlite3 import connect
from spacy.en import English
parser = English()
from sklearn.grid_search import GridSearchCV

from multiprocessing import Pool
from textblob import TextBlob


import nltk.sentiment as sentiment
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.preprocessing import Normalizer

from sklearn.cluster import AgglomerativeClustering, MeanShift
from sklearn.cluster import AffinityPropagation, DBSCAN

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_samples, silhouette_score


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
#parser = English()
# A custom stoplist
STOPLIST = set(stopwords.words('english') + 
               ["message",  'newline','enron','ns','am', 'pm', 'cid','message', 'original', 'from', 'sent','to', 'subject'] +
               list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(" ")

print('load data...')

conn = connect('../hrcemail.sqlite')
c = conn.cursor()

c.execute('SELECT d."docID", n.commonName, d."docDate", d."from", d."to", d.subject, d."docText" FROM document d JOIN name n ON d."from"=n.originalName WHERE n.commonName="Hillary Clinton"')
df_document = pd.DataFrame(c.fetchall(), columns=[h for h,_ ,_ ,_ ,_ ,_ ,_  in c.description])
c.close()
conn.close()
df_document.docDate = pd.to_datetime(df_document.docDate, format='%Y-%m-%d')

def clean_email(email, subject):
    
    start = email.lower().find(subject.lower())
    if start > 0:
        #start += len(subject)
        email = email[start:]
    else:
        start = email.lower().find('Subject')
        if start > 0:
            start += len(subject)
            email = email[start:]
    
    email = email.replace('UNCLASSIFIED U.S. Department of State Case No',"")
          
    body = parser(email)

    tmp_body = []
    for tok in body:
        if (not tok.is_oov):
            if not tok.is_space:
                tmp_body.append(tok.orth_)

    email = ' '.join(tmp_body)
   
        
    return(email)

def tokenizeText(sample):

    # get the tokens using spaCy
    tokens = parser(sample.lower())

    # lemmatize
    tokens = [tok.lemma_.strip() for tok in tokens if tok.is_alpha]
    
    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
#     while "" in tokens:
#         tokens.remove("")
#     while " " in tokens:
#         tokens.remove(" ")
#     while "\n" in tokens:
#         tokens.remove("\n")
#     while "\n\n" in tokens:
#         tokens.remove("\n\n")

    return tokens


print('clean email body...')

df_document['docCleanText'] = [clean_email(e,s) for e, s in zip(df_document.docText.values, df_document.subject.values)]
df_document['Polarity'] = df_document.docCleanText.apply(
    lambda x: TextBlob(x).polarity) 

print('Start model')
vectorizer = TfidfVectorizer(tokenizer=tokenizeText, ngram_range=(1,1), stop_words=STOPLIST)

svd = TruncatedSVD(n_components=100)
pipe_trans = make_pipeline(vectorizer, svd)
dmat = pairwise_distances(pipe_trans.fit_transform(df_document.docCleanText.values),
                          metric='cosine', n_jobs=-1)

print('pre-calculete distance matric..')
param = {'eps': [0.13,  0.18, 0.2, 0.24 ], 'min_samples':[4, 5, 6]}
model = DBSCAN(metric='precomputed')


#X = lsa.fit_transform(df_document.docCleanText.values)

for eps in param['eps']:
    for min_samples in param['min_samples']:
        model.set_params(eps=eps, min_samples= min_samples)
        model.fit(dmat)

        cluster_labels = model.labels_
        try:
            silhouette_avg = silhouette_score(dmat, cluster_labels, metric='precomputed', sample_size=100)
            print('eps: %4.4f, min_samples: %d, NClusters %d' % (eps, min_samples, np.unique(cluster_labels).shape[0]))
            print('silhouette_avg: %4.4' % silhouette_avg)
        except:
            pass




#sample_silhouette_values = silhouette_samples(X.toarray(), cluster_labels)
#print(np.unique(sample_silhouette_values))

print('Done!')
