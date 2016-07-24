import numpy as np
import pandas as pd
from sqlite3 import connect
from spacy.en import English
from textblob import TextBlob
parser = English()
import re

from multiprocessing import Pool
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import string

from nltk.corpus import stopwords
from string import punctuation

from sklearn.pipeline import Pipeline, make_pipeline


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
                'thursday','friday','january','f','message', 'fax','call',
                 'pls', 'fyi', 'start', 'tomorrow','noon'] +
                list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = set(" ".join(punctuation).split(" "))

STOP_UNION = STOPLIST | SYMBOLS | WEEK_MONTH


conn = connect('../data/hrc_emails/hrcemail.sqlite')
# c = conn.cursor()
df_document =pd.read_sql_query("""SELECT * FROM document""",
                   conn)
# c.close()
conn.close()


# df_document = df_document.sample(500)

print('Cleaning emails')
def clean_email(email):
    i0 = email.find('\n')
    email = email.replace(email[:i0], "")
    email = email.replace('RELEASE IN PART',"")
    email = email.replace('RELEASE IN FULL',"")
    email = email.replace('Sent from my iPhone', " ")
    email = email.replace('Sent from my BlackBerry', " ")
    email = email.replace('\n'," ")
    email = email.replace('From:'," ")
    email = email.replace('Sent:'," ")
    email = email.replace('Subject:'," ")

    
    return(email)

# def clean_email(email):
    
#     i0 = email.find('\n')

#     if i0 >0:
#         out =  "|".join([email[:i0], 'RELEASE IN PART', 'RELEASE IN FULL', 
#                   'RELEASE IN FULL','Sent from my iPhone', 'Sent from my iPhone',
#                   'Sent from my BlackBerry', '\n', 'w/','From:', 'Sent:', 'Subject:'])
#     else:
#         out = "|".join([ 'RELEASE IN PART', 'RELEASE IN FULL', 
#                   'RELEASE IN FULL','Sent from my iPhone', 'Sent from my iPhone',
#                   'Sent from my BlackBerry', '\n', 'w/','From:', 'Sent:', 'Subject:'])



#     email = re.sub(out, " ", email)
#     email = re.sub('PRODUCED TO HOUSE SELECT BENGHAZI COMM. SUBJECT TO AGREEMENT ON SENSITIVE INFORMATION',
#                   " ", email)
    
    
#     return(email)



def tokenizeText(sample):

    # get the tokens using spaCy
    tokens = parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        if tok.is_alpha:
            if  (not tok.is_stop):
                lemmas.append(tok.lower_)

    tokens = [tok for tok in lemmas if tok not in STOP_UNION]
    return tokens

df_document['docCleanText'] =  df_document.docText.apply(clean_email)
#get pdfs paths
df_document['pdf_path'] = df_document.docID.apply(lambda x: "".join(["static/pdfs/",x,".pdf"]))


print('Building Tranform Model...')
vectorizer = TfidfVectorizer(tokenizer=tokenizeText, min_df=5,
    ngram_range=(1,2),sublinear_tf=True, use_idf=True,
    stop_words=STOP_UNION)

vectorizer.fit(df_document.docCleanText.values);


print('Extracting main features from each email...')
def get_main_features(X):
    vecX = vectorizer.transform([X])
    names = features_names
    mask = vecX.toarray().argsort(kind='heapsort')[0,::-1]

    return('; '.join(names[mask[:15]]))

features_names = np.array(vectorizer.get_feature_names())
df_document['Keywords'] = df_document.docCleanText.apply(get_main_features)

print('Extracting Sentiment...')
def get_sentiment(X):
    return(TextBlob(X).sentiment)

df_document['Sentiment'] = df_document.docCleanText.apply(get_sentiment)

df_document['Polarity'] = df_document.Sentiment.apply(lambda x: x.polarity)
df_document['Subjectivity'] = df_document.Sentiment.apply(lambda x: x.subjectivity)

df_document.drop('Sentiment', inplace=True, axis=1) 
# print(df_document.info())
# print(df_document.head(8))
# print(df_document.docCleanText.iloc[8])
# print(parser(df_document.docText.iloc[8]))


# Create add columns to docments
# c.execute("""ALTER TABLE document ADD COLUMN docCleanText TEXT""")

# sql=("""UPDATE results SET '
#     + ', '.join(key+' = ?' for key in keys)
#     + 'WHERE id = ?""")

# results = df_document.docCleanText.values
# keys = list(range(results.shape[0]))
# args = [results[key] for key in keys] + [df_document.docID.values]
# c.execute(sql,args)
print(df_document.head(100))

print(df_document.head(50))
conn = connect('../data/hrc_emails/hrcemail2.sqlite')
df_document.to_sql('document',con=conn, if_exists ='replace', index =False)

sql = """SELECT * FROM document;"""
df_document = pd.read_sql_query(sql,conn)
print(df_document.head(20))
conn.close()

