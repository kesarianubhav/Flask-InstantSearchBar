import json
import os

from flask import request, Response
from flask import render_template

from main import app
from main import t_firstName, t_middleName, t_surName, t_Name

from wtforms import Form, StringField, SelectField
from autocomplete_logic import get_best_matches
from autocomplete_logic import populate_Trie
from autocomplete_logic import loadData

import time


class SearchForm(Form):
    choices = [('Firstname', 'FirstName'),
               ('Lastname', 'Lastname'),
               ('Middlename', 'Middlename'),
               ('Name', 'Name')]
    select = SelectField('Search for Name:', choices=choices)
    search = StringField('')


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
    # results = get_best_matches(string, search_category)
    # if search_category == 'Firstname':
    results = get_best_matches(t_firstName, string)
    print(results)
    print(string)
    print(search_category)
    return render_template('index.html', form=search, results=results)


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'
