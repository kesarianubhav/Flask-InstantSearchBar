import json
import os

from flask import request, Response
from werkzeug.utils import secure_filename
from flask import render_template

from main import app
from tasks import *
from utils import random_string
# from forms import SearchForm
from wtforms import Form, StringField, SelectField
from logic import get_best_matches


import time
from time import time
from data_downloader import *
import redis


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


@app.route('/', methods=['GET', 'POST'])
def search_results(search):
    results = []
    string = search.data['search']
    search_category = search.data['select']
    results = get_best_matches(string, search_category)
    print(string)
    print(search_category)
    return render_template('index.html', form=search)


@app.route("/outliers/detect", methods=['POST'])
def outliers_detect():

    bucket = connect_to_bucket()

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            results = {'status': 'failure', 'message': 'No file part'}
            return Response(response=json.dumps(results), status=200, mimetype='application/json')

        uploaded_file = request.files['file']

        # if user does not select file, browser also submit an empty part
        # without filename
        if uploaded_file.filename == '':
            results = {'status': 'failure', 'message': 'No selected file'}
            return Response(response=json.dumps(results), status=200, mimetype='application/json')

        if uploaded_file and allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            uploaded_file.save(file_path)

            file_url = data_uploader(bucket, "server_files", file_path)

            args = {
                'file_path_url': file_url,
                'outlier_space': {'model': 'DBSCAN',
                                  'param':
                                  {
                                           'min_samples': 5,
                                           'eps': 10,
                                           'algorithm': ['auto', 'euclidean'],
                                           'p': 'None',
                                           'leaf_size': 30

                                  }}
            }

            # args = [{'file_path_url': file_url}, ]

            # args

            # result = detect_outliers.delay(file_url, )

            results = task_manager(detect_outliers, args)

            print(results.wait())

            results = {'status': 'success',
                       'message': 'File uploaded successfully'}
            return Response(response=json.dumps(results), status=200, mimetype='application/json')
        else:
            results = {'status': 'failure',
                       'message': 'Invalid file extension'}
            return Response(response=json.dumps(results), status=200, mimetype='application/json')

    else:
        results = {'status': 'failure', 'message': 'Invalid request'}
        return Response(response=json.dumps(results), status=200, mimetype='application/json')


@app.route('/status')
def status():
    data = {}

    response = Response(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )

    return response


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'
