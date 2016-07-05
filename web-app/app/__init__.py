from flask import Flask
from sklearn.externals import joblib
import sys
import sys
sys.path.insert(0,'/home/gondin/metis/project/clinton-email-clusters/')
from search_function import searcher


app = Flask(__name__)
app.config.from_object("app.config")

term_search = 'white house'
# unpickle my model
#estimator = joblib.load('models/iris_model.pkl')
#term_search = ['Ssetosa', 'Sversicolor', 'Svirginica']

#searcher = SearchCluster(X=doc_trans,pipe_trans=pipe_trans, 
#    cluster_model=cluster_model, matrix_similatiry=dmat)

#searcher.fit(new='white house')
        


from .views import *


# Handle Bad Requests
@app.errorhandler(404)
def page_not_found(e):
    """Page Not Found"""
    return render_template('404.html'), 404
