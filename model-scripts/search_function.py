import numpy as np
import pandas as pd
from sqlite3 import connect
from sklearn.externals import joblib
#from nltk.corpus import stopwords

# from spacy.en import English
# parser = English()


# from nltk.corpus import stopwords
# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.decomposition import NMF, TruncatedSVD
# from string import punctuation
# from sklearn.cluster import DBSCAN


from sklearn.metrics.pairwise import cosine_similarity
class SearchCluster:
    def __init__(self, multiplier=3):

        

        #pickle models
        print('loading models...')
        path = '/home/gondin/metis/project/clinton-email-clusters'

        #vectorizer = joblib.load(path + '/vectorizer.pkl')
        #svd = joblib.load(path + '/svd.pkl')
        self.pipe_trans = joblib.load(path + '/pipe_trans.pkl')
        #dmat = joblib.load(path + '/matrix_similatiry.pkl')
        #self.cluster_model = joblib.load(path + '/dbscan_model.pkl')



        print('loading data...')
        # doc_trans = pipe_trans.transform(df_document.docCleanText.values)
        self.doc_trans = joblib.load(path + '/doc_trans.pkl')

        self.multiplier = multiplier

        self.label_ = None
        self.labels_ = None
        self.similarity = None
        #self.vocabulary = joblib.load(path + '/vocabulary.pkl')
        self.model_word2vec = joblib.load(path + '/model_word2vec.pkl')

        

    def most_similar(self, sentence):
        tmp = [(sentence + ' ') * 3]
        sentence = sentence.split(' ')
        
        try:
            l = self.model_word2vec.most_similar(sentence)
            return([word for word, rate in l if rate > 0.9] + tmp)
        except:
            sentence = set(sentence) & self.vocabulary
            if len(sentence) > 0:
                l = model_word2vec.most_similar(sentence)
                return([word for word, rate in l if rate > 0.9] + tmp)
            else:
                return(tmp)


       
        
    def fit(self, new=None, multiplier=3):
        self.multiplier = multiplier
        #new = (new + ' ') * self.multiplier
        
        #self.new = " ".join(self.most_similar(new))
        self.new = [(new + ' ') * 3]

        print(self.new)
        self.new_vec = self.pipe_trans.transform(self.new)
        self.similarity  = 1 - cosine_similarity(self.doc_trans, self.new_vec)
        """
        matrix_similatiry = np.insert(self.matrix_similatiry, -1, newd.T,
                                        axis=1)
        newd = np.append(newd, 0)
        matrix_similatiry = np.insert(matrix_similatiry,-1, newd,
                                        axis=0)
        
        #print('fit preditc')
        labels = self.cluster_model.fit(matrix_similatiry).labels_
        #labels = newd < newd.std()*1.5

        self.label_ = labels[-1]
        self.labels_ = labels[:-1]
        self.similarity = newd
        self.data['dist'] = newd[:-1]
        """
        print(self.similarity.shape)
        # self.data['dist'] = self.similarity[:,0] 

        self.labels_ = (self.similarity  < 0.9)[:,0]
    
        return(self)

    def get_data(self,labels_=None, data=None):
        print('Loading CleanText from DataBase from...')
        conn = connect('/home/gondin/metis/project/clinton-email-download/hrcemail3.sqlite')
        sql = """SELECT Keywords, Polarity, Subjectivity, "from", cluster_labels, pdf_path as "Email" FROM document;"""
        self.data = pd.read_sql_query(sql, conn)
        self.data['Similarity'] = self.similarity[:,0] 

        conn.close()
        
        #self.data = self.data.sample(1000)
        self.data = self.data.sample(15000,random_state=44)


        # labels_ =self.labels_ ==self.label_
        labels_ = self.labels_
        print(self.data.shape)
        print(labels_.shape)
        self.data.Polarity = self.data.Polarity.apply(lambda x: round(x,2)) 



        return (self.data.ix[labels_ & (self.data.cluster_labels>0), ['Keywords','Similarity','Polarity', 'Subjectivity', "from","Email"]].sort_values('Similarity'))
        #return (self.data.ix[labels_, ['Keywords','dist', "Email"]].sort_values('dist'))





print('Create search function')
searcher = SearchCluster() 
"""
new = "white house"
while(new !='exit'):
    new = input('Please, enter the work or sentence or exit to close:')
    if new!='exit':
        searcher.fit(new=new)
        labels = searcher.labels_
        label = searcher.label_
        dist = searcher.similarity

        #print('labels \n', labels)
        #df_document['dist'] = dist[:-1]
        #print(searcher.get_data(labels_=labels, data=df_document))
        print(searcher.get_data())
    else:
        print("finish!")

"""
