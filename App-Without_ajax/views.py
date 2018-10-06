import json
import os
import time
from time import time
import math
import operator

from flask import request, Response
from flask import render_template

from main import app
from main import t_firstName, t_middleName, t_lastName, t_Name

from wtforms import Form, StringField, SelectField, IntegerField
from autocomplete_logic import get_best_matches
from utils import jaccard_score

EPSILON = 1e-4


class SearchForm(Form):
    choices = [('Firstname', 'FirstName'),
               ('Lastname', 'Lastname'),
               ('Middlename', 'Middlename'),
               ('Name', 'Name')]
    select = SelectField('Search for Name:', choices=choices)
    search = StringField('')
    no_of_suggestions = IntegerField('No_of_suggestions')


@app.route('/', methods=['GET', 'POST'])
def start():
    print("Serve is pinged!! Congrats ")
    search = SearchForm(request.form)
    if request.method == 'POST':
        return search_results(search)
    return render_template('index.html', form=search)


# @app.route('/check', methods=['GET', 'POST'])
# def check():
#     print("t_firstName's search:" + str(t_firstName.search('Mahjabeen')))
#     return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def search_results(search):
    string = search.data['search']
    search_category = search.data['select']
    no_of_suggestions = search.data['no_of_suggestions']
    t1 = time()
    results = []
    if len(string) >= 3:
        if search_category == 'Firstname':
            results = get_best_matches(
                t_firstName, string, no=no_of_suggestions)
        if search_category == 'Lastname':
            results = get_best_matches(
                t_lastName, string, no=no_of_suggestions)
        if search_category == 'Middlename':
            results = get_best_matches(
                t_middleName, string, no=no_of_suggestions)
        if search_category == 'Name':
            results = get_best_matches(t_Name, string, no=no_of_suggestions)
    t2 = time()

    jaccards = {x: float(jaccard_score(string, x)) for x in results}
    sorted_jaccards = sorted(
        jaccards.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_jaccards)
    results = sorted_jaccards

    return render_template('index.html', form=search, results=results, querytime=(t2 - t1) * 1000)


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'
