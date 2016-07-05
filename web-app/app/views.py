
from flask import render_template, redirect
from flask_wtf import Form, html5
from wtforms import fields,StringField, TextField
from wtforms.validators import Required
import numpy as np
import pandas as pd

from . import app, searcher, term_search


class PredictForm(Form):
    """Fields for Predict"""
    myChoices = ["one", "two", "three"]
    term_search = TextField('Sentence:', validators=[Required()])
    #term_search = fields.DecimalField('Work:', places='hillary Clinton', validators=[Required()])
    
    #sepal_width = fields.DecimalField('Sepal Width:', places=2, validators=[Required()])
    #petal_length = fields.DecimalField('Petal Length:', places=2, validators=[Required()])
    #petal_width = fields.DecimalField('Petal Width:', places=2, validators=[Required()])

    submit = fields.SubmitField('Search')


@app.route('/', methods=('GET', 'POST'))
def index():
    """Index page"""
    form = PredictForm()
    prediction = None

    if form.validate_on_submit():
        # store the submitted values
        submitted_data = form.data
        print(submitted_data)

        # Retrieve values from form
        term_search = submitted_data['term_search']
        #sepal_width = submitted_data['sepal_width']
        #petal_length = submitted_data['petal_length']
        #petal_width = submitted_data['petal_width']

        # Create array from values
        #flower_instance = [sepal_length, sepal_width, petal_length, petal_width]

        searcher.fit(term_search);
        labels = searcher.labels_
        label = searcher.label_
        dist = searcher.similarity
        prediction = [label]
        df = searcher.get_data()
        pd.options.display.precision = 2
        pd.options.display.max_colwidth = 100
        #df = pd.DataFrame({'column1':[0,1,2], 'column': ['a', 'b', 'c']})
        link_formatter = lambda val: '<a href="%s" target="_blank">Email PDF</a>' % val
        df_html = df.to_html(index=False, 
            classes="u-full-width",
            formatters={
                'Email': link_formatter
            },
            escape=False
            ).replace('border="1"', 'border="0"')
        # Return only the Predicted iris species
        #prediction = target_names[my_prediction].capitalize()
        return render_template('results.html',form=form, prediction='differ', df=df_html)
        

    return render_template('index.html', form=form, prediction=prediction)
