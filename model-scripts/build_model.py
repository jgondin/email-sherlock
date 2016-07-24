import numpy as np
import pandas as pd
from sqlite3 import connect
from sklearn.externals import joblib

from gensim.models import word2vec
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import DBSCAN

from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.metrics import silhouette_samples, silhouette_score


from sklearn.pipeline import Pipeline, make_pipeline
from nltk.corpus import stopwords
from string import punctuation


# A custom stoplist
WEEK_MONTH =set(['monday','tuesday','wednesday,',
                'thursday','friday','january','saturday', 'sunday',
                'fri', 'wed', 'thu', 'thx','ths','pls','ok', 'okay'
                'doc','no',
                'jan','feb', 'mar','apr','may','jun','jul','aug','sep',
                'oct','nov','dec','january', 'february', 'march', 'april', 'may', 'june', 'july',
                'august', 'september', 'october', 'november', 'december','schedule','schedule',
                ])

STOPLIST = set(stopwords.words('english') + 
               ['newline','enron','ns','am', 'pm', 'cid','meeting',
                'original', 'from', 'sent','send','to', 'cc', 'Cc', 'zz',
                'meetings','subject', 'monday','tuesday','wednesday',
                'thursday','friday','january','message', 'fax','call',
                 'pls', 'fyi', 'start', 'tomorrow','noon'] +
                list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = set(" ".join(punctuation).split(" "))

STOP_UNION = STOPLIST | SYMBOLS | WEEK_MONTH


print('Loading data...')
conn = connect('../data/hrc_emails/hrcemail2.sqlite')

sql = """SELECT docCleanText FROM document;"""
df_document = pd.read_sql_query(sql,conn)
conn.close()

df_document = df_document.sample(15000,random_state=44)


#pickle path 
path = '/home/gondin/metis/project/clinton-email-clusters'


print('Building Tranform Model...')

from tokenizer import tokenizeText

vectorizer = TfidfVectorizer(tokenizer=tokenizeText, min_df=3,
	ngram_range=(1,3), sublinear_tf=True, use_idf=True,
	stop_words=STOP_UNION)
svd = TruncatedSVD(n_components=125)
pipe_trans = make_pipeline(vectorizer, svd)


print('Transforming imput text...')
pipe_trans = pipe_trans.fit(df_document.docCleanText.values)

print('Calculating Cosine Similarities...')
m = df_document.docCleanText.values.shape[0]
mid = m // 2

X = pipe_trans.transform(df_document.docCleanText.values)
matrix_similatiry = pairwise_distances(X[:mid], metric='cosine', n_jobs=4)
print(X.shape)

for i in range(mid,m):
    #X = pipe_trans.transform(df_document.docCleanText.values[:i])
    # new_vec = pipe_trans.transform(
    #     [df_document.docCleanText.values[i]])
    
    new_vec = X[i,:]
    new_vec.resize(1,new_vec.shape[0])
    
    newd  = 1 - cosine_similarity(X[:i,:], new_vec)
    #X = np.insert(X, X.shape[0], new_vec, axis=0)
    
    matrix_similatiry = np.insert(matrix_similatiry,
        matrix_similatiry.shape[1],
        newd.T, axis=1)
    
    newd = np.append(newd, 1.0)
    
    matrix_similatiry = np.insert(matrix_similatiry,
        matrix_similatiry.shape[0],
        newd, axis=0)        


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




doc_trans = pipe_trans.transform(df_document.docCleanText.values)

# #dmat = 1 - cosine_similarity(doc_trans)
# dmat = pairwise_distances(doc_trans, metric='cosine', n_jobs=3)

print('Pickle doc_trans...')
joblib.dump(doc_trans,
	path + '/doc_trans.pkl')
doc_trans = None

print('Pickle pipe_trans')
joblib.dump(pipe_trans,
	path + '/pipe_trans.pkl')
vectorizer = None
svd = None
pipe_trans = None

dmat = matrix_similatiry
del matrix_similatiry

print('Building Clustering Model...')
s = dmat.std()
s *= 2
model = DBSCAN(eps=s, min_samples=4, metric='precomputed')

print('Fitting Clustering Model...')
model.fit(dmat)
cluster_labels = model.labels_
df_document['cluster_labels'] = cluster_labels
print('Number of cluster %d' % np.unique(cluster_labels).shape[0])

print('Pickle matrix_similarity...')
joblib.dump(dmat,
	path + '/matrix_similatiry.pkl')
dmat = None

print('Pickle Model...')
joblib.dump(model,
	path + '/dbscan_model.pkl')



print('Building work2vec...')

sentences = df_document.docCleanText.apply(tokenizeText).values
model_word2vec = word2vec.Word2Vec(sentences,size=200, window=10,
	min_count=3)

vocabulary = set([])
for x in sentences:
	vocabulary |= set(x)
    
df_document['email_len'] =[len(x) for x in enumerate(sentences)]



#pickle models
joblib.dump(vocabulary,
	path + '/vocabulary.pkl')
joblib.dump(model_word2vec,
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


# print('Start search function')
# def search_cluster(new=None,dmat=None, pipe_trans=None):
#     new = np.array([new])
#     new_vec = pipe_trans.transform(new)
    
#     newd = 1 - cosine_similarity(doc_trans, new_vec)
    
# #     tmp = np.insert(dmat, n, newd, axis=1)
# #     tmp = np.insert(tmp , n, np.append(newd, 0), axis=0)
    
#     dmat = np.insert(dmat, -1, newd.T,axis=1)
#     newd = np.append(newd, 0)
#     dmat = np.insert(dmat,-1, newd, axis=0)
    
#     model.fit(dmat)
#     labels = model.labels_
#     label = labels[-1]
#     labels = labels[:-1]

#     if label == -1:
#     	s = dmat.std()
#     	model.set_params(eps=s*1.2)
#     	model.fit(dmat)
#     	labels = model.labels_
#     	label = labels[-1]
#     	labels = labels[:-1]



   
#     return (labels, label, newd)
# flag = ""
# while(flag !='exit'):
# 	new = input('Please, enter the work or sentence or exit to close:')
# 	new = (new + ' ')*2
# 	#new = ('benghazi' + ' ' )*2
# 	(labels, label, dist) = search_cluster(new=new, dmat=dmat, pipe_trans=pipe_trans)


# 	#print('labels \n', labels)
# 	print(labels.shape)
# 	print(df_document.shape)
# 	print(dist.shape)
# 	df_document['dist'] = dist[:-1]
# 	print(df_document.ix[labels==label,:].sort_values('dist'))


















