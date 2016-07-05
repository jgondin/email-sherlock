#!/usr/bin/env python
import numpy as np
import pandas as pd
from sqlite3 import connect
from sklearn.externals import joblib

from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from string import punctuation
from sklearn.cluster import DBSCAN

from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.metrics import silhouette_samples, silhouette_score


from sklearn.pipeline import Pipeline, make_pipeline
from spacy import English

parser = English()

from multiprocessing import Pool
from textblob import TextBlob

# A custom stoplist
STOPLIST = set(stopwords.words('english') + 
               ['newline','enron','ns','am', 'pm', 'cid',
                'original', 'from', 'sent','to', 'subject', 'cc', 'Cc', 'zz'] +
               list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = set(" ".join(punctuation).split(" "))

print('functions')
def clean_email(email, subject):
    email = email.replace('UNCLASSIFIED U.S. Department of State Case No',"")
    
    email = email.lower()
    subject = subject.lower()
    start = email.find(subject)
    if start > 0:
        #start += len(subject)
        email = email[start:]
    else:
        start = email.find('Subject')
        if start > 0:
            start += len(subject)
            email = email[start:]
    
          
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



print('Loading data...')
conn = connect('../clinton-email-cruncher/hrcemail3.sqlite')
sql = """SELECT docText FROM document;"""
df_document = pd.read_sql_query(sql,conn)
conn.close()

df_document = df_document.sample(500)
df_document['docCleanText'] =df_document.docText


print('Building Tranform Model...')
vectorizer = TfidfVectorizer( ngram_range=(1,3), stop_words=(STOPLIST | SYMBOLS))
svd = TruncatedSVD(n_components=110)
pipe_trans = make_pipeline(vectorizer, svd)


print('Transforming imput text...')
doc_trans = pipe_trans.fit_transform(df_document.docCleanText.values)

print('Calculating Cosine Similarities...')

#dmat = 1 - cosine_similarity(doc_trans)
dmat = pairwise_distances(doc_trans,metric='cosine', n_jobs=3)


print('Building Clustering Model...')
s = dmat.std()
s *= 2
model = DBSCAN(eps=s, min_samples=4, metric='precomputed',
    algorithm='auto')

print('Fitting Clustering Model...')
model.fit(dmat)
cluster_labels = model.labels_


print('Run optmazation...')
param = {'eps': [ 0.75,  1, 1.5, 2, 0.25, 3 ],
         'min_samples':[2, 3, 4, 5, 7],
         'ngram_range': [(1,1), (1,2), (1,3), (1,4)],
         'n_components':[ 100,  125, 150 ],
         'tokenizer':[None, tokenizeText]
        }

def scores(dmat,cluster_labels):
    try:
        silhouette_avg = silhouette_score(dmat, cluster_labels, metric='precomputed', sample_size=100)
        return(silhouette_avg)
    except:
        return(None)
epsN = []
min_samplesN =[]
ngram_rangeN = []
n_componentsN = []
tokenizerN = []
silhouette_avgN = []
sigma = []
n_clusters =[]


for ngram_range in param['ngram_range']:
    
    for tokenizer in param['tokenizer']:
        vectorizer.set_params(ngram_range=ngram_range, tokenizer= tokenizer,
                              stop_words=(STOPLIST | SYMBOLS))
        for n_components in param['n_components']:
            svd.set_params(n_components=n_components)
            
            X_trans = pipe_trans.fit_transform(df_document.docCleanText.values)
            dmat = 1 - cosine_similarity(X_trans)
            s = dmat.std()


            for eps in param['eps']:
                for min_samples in param['min_samples']:
                    model.set_params(eps=eps*s, min_samples= min_samples)
                    model.fit(dmat)

                    cluster_labels = model.labels_
                    n_clusters.append(np.unique(cluster_labels).shape[0])
                    silhouette_avg = scores(dmat,cluster_labels)
                    epsN.append(eps)
                    min_samplesN.append(min_samples)
                    ngram_rangeN.append(ngram_range)
                    n_componentsN.append(n_components)
                    tokenizerN.append(tokenizer)
                    silhouette_avgN.append(silhouette_avg)
                    sigma.append(s)
                    #print('okay')

df = pd.DataFrame({'epsN':epsN,
                    'min_samplesN':min_samplesN,
                    'ngram_rangeN':ngram_rangeN,
                    'n_componentsN':n_componentsN,
                    'tokenizerN':tokenizerN,
                    'silhouette_avgN':silhouette_avgN,
                    'sigma':sigma,
                    'n_clusters':n_clusters
                    })
df.to_csv('opt.csv')
print(df)