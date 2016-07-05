from string import punctuation
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from spacy.en import English


parser = English()

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


def tokenizeText(sample):

    # get the tokens using spaCy
    tokens = parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        if tok.is_alpha:
            if (not tok.is_stop):
                lemmas.append(tok.lower_)

    tokens = [tok for tok in lemmas if tok not in STOP_UNION]
    return tokens
