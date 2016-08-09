import numpy as np
import pandas as pd
from sqlite3 import connect
from gensim.models import word2vec

from sklearn.externals.joblib import dump
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity, pairwise_distances
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer

from tokenizer import tokenizeText



print('Loading data...')

conn = connect('../data/hrc_emails/hrcemail2.sqlite')

sql = """SELECT docCleanText FROM document;"""
df_document = pd.read_sql_query(sql,conn)
conn.close()

df_document = df_document.sample(15000,random_state=44)


#pickle path 
path = '../data/hrc_emails/pickle'


print('Building Tranform Model...')


vectorizer = TfidfVectorizer(tokenizer=tokenizeText, min_df=3,
	ngram_range=(1, 3), sublinear_tf=True, use_idf=True)

svd = TruncatedSVD(n_components=150)
normalizer = Normalizer()
pipe_trans = make_pipeline(vectorizer, svd, normalizer)


print('Transforming imput text...')
X = pipe_trans = pipe_trans.fit_transform(df_document.docCleanText.values)

print('Calculating Cosine Similarities...')

# X = pipe_trans.transform(df_document.docCleanText.values)
# m = X.shape[0]
# mid = m // 2

matrix_similatiry = linear_kernel(X)
print(matrix_similatiry.shape)


# for i in range(mid,m):
#     #X = pipe_trans.transform(df_document.docCleanText.values[:i])
#     # new_vec = pipe_trans.transform(
#     #     [df_document.docCleanText.values[i]])
    
#     new_vec = X[i,:]
#     new_vec.resize(1,new_vec.shape[0])
    
#     newd  = 1 - cosine_similarity(X[:i,:], new_vec)
#     #X = np.insert(X, X.shape[0], new_vec, axis=0)
    
#     matrix_similatiry = np.insert(matrix_similatiry,
#         matrix_similatiry.shape[1],
#         newd.T, axis=1)
    
#     newd = np.append(newd, 1.0)
    
#     matrix_similatiry = np.insert(matrix_similatiry,
#         matrix_similatiry.shape[0],
#         newd, axis=0)        

# X = pipe_trans.transform(df_document.docCleanText.values)
# dmat3 = 1 - cosine_similarity(X)
# print('shapes:')
# print(matrix_similatiry.shape,dmat3.shape)
# print(matrix_similatiry is dmat3)
# print('matrix_similatiry == dmat3:',sum(sum(matrix_similatiry != dmat3)))
# d = matrix_similatiry - dmat3
# print(matrix_similatiry-dmat3)
# d.resize(1,d.shape[0])
# print(np.sum(d**2))




#doc_trans = pipe_trans.transform(df_document.docCleanText.values)

# #dmat = 1 - cosine_similarity(doc_trans)
# dmat = pairwise_distances(doc_trans, metric='cosine', n_jobs=3)

print('Pickle doc_trans...')
dump(X,
	path + '/doc_trans.pkl')
X = None

print('Pickle pipe_trans')
dump(pipe_trans,
	path + '/pipe_trans.pkl')

vectorizer = None
svd = None
pipe_trans = None


print('Building Clustering Model...')
s = matrix_similatiry.std()
s *= 2
model = DBSCAN(eps=s, min_samples=4, metric='precomputed')

print('Fitting Clustering Model...')
model.fit(matrix_similatiry)
cluster_labels = model.labels_
df_document['cluster_labels'] = cluster_labels

print('Number of cluster %d' % np.unique(cluster_labels).shape[0])

print('Pickle matrix_similarity...')
dump(matrix_similatiry,
	path + '/matrix_similatiry.pkl')
matrix_similatiry = None

print('Pickle Model...')
dump(model,
	path + '/dbscan_model.pkl')


print('Building work2vec...')

sentences = df_document.docCleanText.apply(tokenizeText).values
model_word2vec = word2vec.Word2Vec(sentences, size=200, window=10,
	min_count=3)

vocabulary = set([])
for x in sentences:
	vocabulary |= set(x)
    
df_document['email_len'] =[len(x) for x in enumerate(sentences)]


#pickle models
dump(vocabulary,
	path + '/vocabulary.pkl')
dump(model_word2vec,
	path + '/model_word2vec.pkl')


print(df_document.head(8))


conn = connect('../data/hrc_emails/hrcemail2.sqlite')

sql = """SELECT * FROM document;"""
df_document_all = pd.read_sql_query(sql,conn)

df_document_all = df_document_all.sample(15000,random_state=44)

conn.close()

df_document_all['cluster_labels'] =cluster_labels
df_document_all['docCleanText'] = df_document['docCleanText']
df_document_all['email_len'] = df_document['email_len'] 



print(df_document_all.head(16))
print(df_document_all.shape)
conn = connect('../data/hrc_emails/hrcemail3.sqlite')
df_document_all.to_sql('document',con=conn, if_exists ='replace', index =False)
df_document_all = None

sql = """SELECT * FROM document;"""
df_document = pd.read_sql_query(sql,conn)
print(df_document.head(20))
print(df_document.shape)
conn.close()


print('Finish!')