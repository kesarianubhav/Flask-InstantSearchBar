# import time
# from time import time
import celery
import pandas as pd
import sklearn
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.feature_selection import SelectKBest, SelectFdr, SelectFpr, SelectFwe
import sklearn
from data_imputation import MiceImputer
import os
from celery.signals import task_success
# from main import db
# from models import db
# from models import training_stats
from utils import random_string
from main import app
from celery import signals
from time import *
from signal import pause
from data_downloader import connect_to_bucket
from data_downloader import get_all_temp_url
from data_downloader import get_temp_url
from data_downloader import data_uploader
# from data_downloader import data_downloader
import signal
import redis
# import celery_pubsub as cp
from celery_pubsub import publish, subscribe
# from celery.task.http import URL
# from celery.task.http import HTMLUrl
# from celery.task.http import URL
# from celery.task.http import HttpDispatchTask

from io import StringIO
from utils import *

from pubsub import pub

import asyncio
import aioredis
from utils import Store
# pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
# r = redis.Redis(connection_pool=pool)


# from production.pipeline.data_imputation import MiceImputer


def init_db(app):

    if app.config['DEBUG']:
        # print 'Recreating all db'
        db.create_all()  # I DO create everything
        # print 'Loading test data'
        # ... (here I add some Users and etc. and everything works fine - tests pass)

    return db


# @celery.task
# def add_callback():
#     return "bhk !!!!"


@signals.task_success.connect
def task_successful(sender, result, **keyargs):

    print("TASK_SUCCESSFUL")

    y_predicted = result[1]
    task_id = result[0]

    print("TASK __>" + str(task_id))

    output_filename = storeData(
        y_predicted, "result" + str(random_string()))

    print(output_filename)

    bucket = connect_to_bucket()
    url = data_uploader(bucket, "server_files", output_filename)

    print(url)

    sender_name = sender.name

    item = training_stats.query.filter_by(t_id=task_id).first()

    # if not len(item) == 0:
    item.t_status = 'COMPLETED'
    item.t_picklepath = str(url)

    db.session.commit()


@signals.task_failure.connect
def task_failure(sender, result, **keyargs):

    print("TASK_FAILED")

    task_id = result[0]
    # print(result)
    print(task_id)

    sender_name = sender.name

    item = training_stats.query.filter_by(t_id=task_id).first()
    if not len(item) == 0:
        item.t_status = 'FAILED'
    db.session.commit()


@signals.after_task_publish.connect
def handle_after_task_publish(sender=None, body=None, **kwargs):

    t_type = str(sender)
    t_time = str(time())
    # try:
    #     print(time())
    # except Exception as e:
    #     print(e)
    t_id = str(body[1]['task_id'])
    t_dataset = str(body[1]['file_path_url'])
    # t_id = kwargs['task_id']
    # print("TYPE " + str((kwargs['headers']['kwargsrepr'])))
    # print("ID" + str((kwargs['headers']['kwargsrepr']['task_id'])))
    # print("BODY H YEH ISKI " + str(body[1]['task_id']))
    # print()
    # print("Adskjgn          bhfs             " + str(t_id))
    # print("body" + str(body[0][0]))
    t_status = "PENDING"
    t_picklepath = "PENDING"
    # t_dataset = "file_path"
    # print("dfvfdv" + str(t_id))

    entry = training_stats(t_type=t_type, t_time=t_time, t_id=t_id,
                           t_status=t_status, t_picklepath=t_picklepath, t_dataset=t_dataset)

    sess = db.session
    sess.add(entry)
    sess.commit()

    # print('DONe')`


@celery.task(name='add')
def add(t_id):

    s1 = Store(store_id=t_id)

    s1.setState('SHURU HO RHA HOON')

    print("STATE=" + str(s1.getState()))
    # signal.pause()
    sleep(5)
    s1.setState('CHAL RHA HOON')
    # self.update_state(state="PROGRESS", meta={'progress': 50})
    print("STATE=" + str(s1.getState()))
    sleep(5)
    s1.setState('BS KHATAM HONE WALA HOON')
    # self.update_state(state="PROGRESS", meta={'progress': 90})
    sleep(5)

    result = [None] * 2
    result[0] = "task_id"
    result[1] = str(1) + str(1)
    # pub.subscribe(check, 'STATUS.add')

    return result


@celery.task(name='detect_outliers')
def detect_outliers(task_id, file_path_url, outlier_space):

    # bucket = connect_to_bucket()
    # file_path = data_downloader(bucket, file_path_url)

    # outlier_space = outlier_space

    url = file_path_url
    # print(
    #     "======================================================================")
    # print("Dataset Downloading for url :" + str(url))
    # print(
    #     "======================================================================")

    # data_frame = pd.read_csv(file_path, sep=",|;",
    #                          header=0, engine='python')

    data_frame = url_to_dataframe(file_path_url)

    Y = data_frame['label']
    data_frame.drop(['label'], axis=1, inplace=True)
    X = data_frame

    y_predicted = None

    # try:

    # ot1 = time()
    ot1 = time()

    print("Outling Value Treatment Started ........................................")
    print("=========================================================================================================================================")

    if outlier_space['model'] == "DBSCAN":
        db = DBSCAN(eps=outlier_space['param']['eps'], min_samples=outlier_space['param']['min_samples'],
                    algorithm=outlier_space['param']['algorithm'][
            0], metric=outlier_space['param']['algorithm'][1],
            p=outlier_space['param']['p'], leaf_size=outlier_space['param']['leaf_size'])
        y_predicted = db.fit_predict(X)

        y_predicted = list(map(lambda x: 1 if x < 0 else 0, y_predicted))
        print("DBSCAN")

    elif outlier_space['model'] == "EllipticEnvelope":
        elliptic = EllipticEnvelope(contamination=outlier_space['param']['contamination'],
                                    assume_centered=outlier_space[
            'param']['assume_centered'],
            support_fraction=outlier_space['param']['support_fraction'])
        elliptic.fit(X)
        y_predicted = elliptic.predict(X)

        y_predicted = list(map(lambda x: 1 if x == -1 else 0, y_predicted))
        print("ELIPTIC_ENVELOPE")

    elif outlier_space['model'] == "IsolationForest":
        iso = IsolationForest(n_estimators=outlier_space['param']['n_estimators'],
                              contamination=outlier_space[
            'param']['contamination'],
            max_features=outlier_space[
            'param']['max_features'],
            max_samples=outlier_space[
            'param']['max_samples'],
            bootstrap=outlier_space['param']['bootstrap'])
        iso.fit(X)
        y_predicted = iso.predict(X)

        y_predicted = list(map(lambda x: 1 if x == -1 else 0, y_predicted))
        print("ISOLATION_FOREST")

    elif outlier_space['model'] == "OneClassSVM":
        ocv = OneClassSVM(kernel=outlier_space['param']['kernel'], degree=outlier_space['param']['degree'],
                          max_iter=outlier_space['param'][
                              'max_iter'], nu=outlier_space['param']['nu'],
                          shrinking=outlier_space['param']['shrinking'],
                          gamma=outlier_space['param']['gamma'])
        ocv.fit(X)
        y_predicted = ocv.predict(X)

        y_predicted = list(map(lambda x: 1 if x == -1 else 0, y_predicted))
        print("ONECLASS_SVM")

    elif outlier_space['model'] == "LocalOutlierFactor":
        lof = LocalOutlierFactor(n_neighbors=outlier_space['param']['n_neighbors'],
                                 contamination=outlier_space[
                                     'param']['contamination'],
                                 leaf_size=outlier_space[
                                     'param']['leaf_size'],
                                 algorithm=outlier_space[
                                     'param']['algorithm'][0],
                                 p=outlier_space['param']['p'], metric=outlier_space['param']['algorithm'][1])
        y_predicted = lof.fit_predict(X)

        y_predicted = list(map(lambda x: 1 if x == -1 else 0, y_predicted))
        print("LOCAL_OUTLIER_FACTOR")

    elif outlier_space['model'] == "zscore":
        threshold = outlier_space['param']['threshold']

        score = zscore(X, axis=0, ddof=1)
        score_frame = pd.DataFrame(data=score, columns=X.columns.values)
        score_frame = score_frame.abs()

        outlier_count = 0

        predicted_outliers = []
        for i in range(len(score_frame)):
            if any(score_frame.iloc[i] > threshold) is True:
                predicted_outliers.append(1)
                outlier_count += 1
            else:
                predicted_outliers.append(0)
        y_predicted = predicted_outliers
        print("ZSCORE")
    # Y_predicted added to original dataframe

    # print(X)
    # print(y_predicted)
    # print(len(y_predicted))
    # return 1

    # print("X KA SHAPE :"+str(X.shape))
    # print("Y KA SHAPE :"+str(Y.shape))

    total_rows = X.shape[0]

    # print(type(y_predicted))

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    X['y_predicted'] = y_predicted
    Y['y_predicted'] = y_predicted

    indexe_outliers = ((X[X['y_predicted'] == 1].index))
    # # print(X.shape)
    # # dropped the index of outliers

    # print(indexe_outliers)
    no_of_outliers = (len(indexe_outliers))
    if not (no_of_outliers == 0 or no_of_outliers == X.shape[0]):
        # if len(indexe_outliers)!=0:
        X = X.drop(X[X['y_predicted'] == 1].index, axis=0)
        Y = Y.drop(Y[Y['y_predicted'] == 1].index, axis=0)

    #     # droped the Y_predicted column to gain original dataframe
    X = X.drop(['y_predicted'], axis=1)
    Y = Y.drop(['y_predicted'], axis=1)

    # shape = X.shape
    print("X and Y s' size " + str(X.shape) + str(Y.shape))

    outlier_extracted_percentage = ((no_of_outliers) / total_rows) * 100

    print("OUTLIER EXTRACTED PERCENTAGE " +
          str(outlier_extracted_percentage))

    print("Outlier Extraction Treatment Done")
    print("=========================================================================================================================================")
    outlier_prediction_time = time() - ot1

    # except Exception as e:
    #     print("Exception Catched in Outlier Preprocessing Pipeline")
    #     print("Exception:" + str(e))
    #     pass

    # print(X.head())

    # pass

    # X['label'] = Y
    # X = pd.DataFrame(X)

    # print("LE BE ITNE HAIN OUTLIER" + str(len(y_predicted)))
    # print(y_predicted)

    # data_uploader(output_file)

    # results = {
    #     'picklepath': output_filename,
    #     'result': y_predicted
    # }

    # output_filename = storeData(outlier_space, "result" + str(random_string()))

    # bucket = connect_to_bucket()
    # url = data_uploader(bucket, "server_files", output_filename)

    # return output_filename
    return [task_id,  y_predicted]


@celery.task(name='auto_detect_outlier')
def auto_detect_outliers(file_path):

    # print(file_path)

    hyperopt_trained_outlier_space = [{'model': 'DBSCAN',
                                       'param':
                                       {
                                           'min_samples': 5,
                                           'eps': 10,
                                           'algorithm': ['auto', 'euclidean'],
                                           'p': 'None',
                                           'leaf_size': 30

                                       }},
                                      {'model': 'DBSCAN',
                                       'param':
                                       {
                                           'min_samples': 5,
                                           'eps': 5,
                                           'algorithm': ['auto', 'euclidean'],
                                           'p': 'None',
                                           'leaf_size': 30

                                       }},
                                      {'model': 'DBSCAN',
                                       'param':
                                       {
                                           'min_samples': 5,
                                           'eps': 0.5,
                                           'algorithm': ['auto', 'euclidean'],
                                           'p': 'None',
                                           'leaf_size': 30

                                       }}
                                      ]
    df = pd.read_csv(file_path, sep=',|;', header=0, engine='python')
    df.fillna(0)

    # print(hyperopt_trained_outlier_space[0]['model'])
    # print(hyperopt_trained_outlier_space[0]['param']['min_samples'])
    # print('')

    all_votes = []

    for i in hyperopt_trained_outlier_space:
        y_predicted = detect_outliers(file_path, i)
        all_votes.append(y_predicted)

    all_votes = np.array(all_votes).reshape(
        (len(y_predicted), len(hyperopt_trained_outlier_space)))

    # print(all_votes.shape)

    all_votes = np.count_nonzero(all_votes, axis=1)
    voting_percents = all_votes / (len(hyperopt_trained_outlier_space)) * 100

    # print(len(voting_percents))

    df_outlier_dropped = df.drop(df.index[voting_percents])

    print(df_outlier_dropped.shape)

    return voting_percents


@celery.task(name='treat_missing_values')
def treat_missing_values(file_path, space_missing_values_treatment):
    df = pd.read_csv(file_path, sep=",|;", header=0, engine='python')

    Y = df['label']
    X = df.drop(['label'], axis=1)

    print("==================================================")
    print("Treating missing values...")
    print("Space:", space_missing_values_treatment)

    # try:
    if space_missing_values_treatment['model'] == "DecisionTreeRegressor":
        criter = space_missing_values_treatment["params"]["criterion"]
        max_feature = space_missing_values_treatment[
            "params"]['max_features']
        reg = sklearn.tree.DecisionTreeRegressor(
            criterion=criter, max_features=max_feature)

    elif space_missing_values_treatment['model'] == "MLPRegressor":
        hidden_layer_sizes = [
            x + 1 for x in space_missing_values_treatment["params"]['hidden_layer_sizes']]
        activation = space_missing_values_treatment["params"]["activation"]
        solver = space_missing_values_treatment["params"]["solver"]
        alpha = space_missing_values_treatment["params"]["alpha"]
        learning_rate = space_missing_values_treatment[
            "params"]["learning_rate"]
        if solver == 'sgd':
            reg = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                                      alpha=alpha, learning_rate=learning_rate)
        else:
            reg = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                                      alpha=alpha)

    elif space_missing_values_treatment['model'] == "KNeighborsRegressor":
        algorithm = space_missing_values_treatment["params"]["algorithm"]
        p = 1 + space_missing_values_treatment["params"]["p"]
        leaf_size = 1 + \
            space_missing_values_treatment["params"]["leaf_size"]
        if leaf_size >= df.shape[1]:
            print("setting leaf_size to max in KNeighborsRegressor")
            leaf_size = df.shape[1] - 1

        if algorithm == "kd_tree" or algorithm == 'ball_tree':
            reg = sklearn.neighbors.KNeighborsRegressor(
                algorithm=algorithm, p=p, n_jobs=-1, leaf_size=leaf_size)
        else:
            reg = sklearn.neighbors.KNeighborsRegressor(
                algorithm=algorithm, p=p, n_jobs=-1)

    elif space_missing_values_treatment['model'] == "GradientBoostingRegressor":
        loss = space_missing_values_treatment["params"]["loss"]
        learning_rate = space_missing_values_treatment[
            "params"]["learning_rate"]
        n_estimators = 1 + \
            space_missing_values_treatment["params"]["n_estimators"]
        max_depth = 1 + \
            space_missing_values_treatment["params"]["max_depth"]
        criterion = space_missing_values_treatment["params"]["criterion"]
        max_features = space_missing_values_treatment[
            "params"]["max_features"]

        # error handling
        if n_estimators >= df.shape[1]:
            print("setting n_estimators to max in GradientBoostingRegressor")
            n_estimators = df.shape[1] - 1

        reg = GradientBoostingRegressor(loss=loss, n_estimators=n_estimators,
                                        learning_rate=learning_rate, criterion=criterion, max_depth=max_depth)
    else:
        print('Missing values failed due to error in space')
        cost = 9999999

    # delete nan data
    per = (X.isnull().sum() / X.isnull().count()).sort_values()
    idx = per[per >= 0.5].index
    X.drop(idx, axis=1, inplace=True)
    pd.DataFrame(Y).drop(idx, axis=1, inplace=True)

    a = MiceImputer()
    try:
        X = a.fit_transform(X, model=reg)
    except ValueError as v:
        print("Error", v)
        print("TODO replace string by mode maybe")

    # except Exception as e:
    #     print("error", e)

    print("Missing values treatment finished")
    print("==================================================\n")

    """
    Return a dataframe or a csv file
    """
    X['label'] = Y
    X.to_csv("missing_treated_file.csv")
    # "a_converted.csv"
    converted_path = str(os.getcwd() + "/missing_treated_file.csv")
    return X


@celery.task(name='auto_treat_missing_values')
def auto_treat_missing_values(file_path):
    missing_space = [{'model': 'DecisionTreeRegressor', 'params': {'max_features': 'log2', 'criterion': 'mse'}},
                     {'model': 'DecisionTreeRegressor', 'params': {
                         'max_features': 'auto', 'criterion': 'friedman_mse'}}

                     ]
    df = pd.read_csv(file_path, sep=',|;', header=0, engine='python')

    df = np.array(df.values)

    print(df)

    for i in missing_space:

        df = df + treat_missing_values(file_path, i).values
    df = df / (len(missing_space) + 1)

    df = pd.DataFrame(df)
    df.to_csv("missing_treated_file.csv")
    # "a_converted.csv"
    converted_path = str(os.getcwd() + "/missing_treated_file.csv")

    print(df)

    return converted_path


@celery.task(name='feature_selection')
def feature_selection(file_path, feature_space):

    print(" Feature Extraction Value Treatment Started")
    print("=========================================================================================================================================")

    # try:
    ft1 = time()
    df = pd.read_csv(file_path, sep=',|;', header=0, engine='python')
    df.fillna(0)
    # global flag

    # print "\n\n\n",args,"\n\n\n"
    Y = df['label']
    X = df.drop(['label'], axis=1)
    features = X.shape[1]

    X = X.values
    Y = Y.values

    flag = time()
    flag = str(flag)
    # print(X.shape, Y.shape)

    Y = Y.reshape((Y.shape[0], 1))

    # print(X.shape, Y.shape)

    dic = {}
    if feature_space['model'] == "SelectKBest":
        score_func = feature_space["params"]["score_func"]
        if score_func == "f_regression":
            score_func = f_regression
        if score_func == "mutual_info_regression":
            score_func = mutual_info_regression
        k = feature_space["params"]["k"]
        new_x = SelectKBest(score_func=score_func, k=k).fit_transform(X, Y)
        feature_extraction_space = {'count': flag, 'model': 'SelectKBest',
                                    'score_func': str(score_func), 'k': k}
        print("SELECTKBEST")

    elif feature_space['model'] == "regression_feature_selector":
        k_f = feature_space["params"]["k_f"]
        k_mi = feature_space["params"]["k_mi"]
        nn_mi = feature_space["params"]["nn_mi"]
        print("nn_mi", nn_mi)
        new_x = custom_selector.regression_feature_selector(
            X, Y, k_f=k_f, k_mi=k_mi, nn_mi=nn_mi)

        feature_extraction_space = {'count': flag, 'model': 'tanay_regression',
                                    'k_f': k_f, 'k_mi': k_mi, 'nn_mi': nn_mi}
        print("REGRESSION_FEATURE_SELECTOR")

    elif feature_space['model'] == "SelectFdr":
        score_func = feature_space['params']['score_func']
        if score_func == "f_regression":
            score_func = f_regression
        if score_func == "mutual_info_regression":
            score_func = mutual_info_regression
        alpha = feature_space['params']['alpha']
        new_x = SelectFdr(score_func=score_func,
                          alpha=alpha).fit_transform(X, Y)
        feature_extraction_space = {'count': flag, 'model': 'SelectFdr',
                                    'score_func': str(score_func), 'alpha': alpha}
        print("SELECTFDR")

    elif feature_space['model'] == "SelectFpr":
        score_func = feature_space['params']['score_func']
        if score_func == "f_regression":
            score_func = f_regression
        if score_func == "mutual_info_regression":
            score_func = mutual_info_regression
        alpha = feature_space['params']['alpha']
        new_x = SelectFpr(score_func=score_func,
                          alpha=alpha).fit_transform(X, Y)
        feature_extraction_space = {'count': flag, 'model': 'SelectFpr',
                                    'score_func': str(score_func), 'alpha': alpha}
        print("SELECTFPR")

    elif feature_space['model'] == "SelectFwe":
        score_func = feature_space['params']['score_func']
        if score_func == "f_regression":
            score_func = f_regression
        if score_func == "mutual_info_regression":
            score_func = mutual_info_regression
        alpha = feature_space['params']['alpha']
        new_x = SelectFwe(score_func=score_func,
                          alpha=alpha).fit_transform(X, Y)
        feature_extraction_space = {'count': flag, 'model': 'SelectFwr',
                                    'score_func': str(score_func), 'alpha': alpha}
        print("SELECTFWE")

    X_frame = pd.DataFrame(X, columns=["attr" + str(i)
                                       for i in range(0, (X.shape[1]))])

    new_x_frame = pd.DataFrame(new_x, columns=["attr" + str(i)
                                               for i in range(0, (new_x.shape[1]))])

    # print(dic)
    X = new_x
    # print("after feature extraction:", X.shape, Y.shape)

    features_extracted = X.shape[1]

    features_extracted_percentage = (features_extracted / features) * 100

    print("FEATURES EXTRACTED PERCENTAGE " +
          str(features_extracted_percentage))
    print("Feature Extraction Done")
    feature_extraction_time = time() - ft1
    print("=========================================================================================================================================")

    """`"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # print(X_frame.shape)
    # print(new_x_frame.shape)

    indexes = []

    for i in range(0, X_frame.shape[1]):
        for j in range(0, new_x_frame.shape[1]):
            if(X_frame[X_frame.columns[i]].equals(new_x_frame[new_x_frame.columns[j]])):
                indexes.append(i)
                break
            # except Exception as e:
            #     print("Exception Catched in Feature Extraction Preprocessing Pipeline")
            #     print("Exception:" + str(e))
            #     pass
    """
    returning the columnns of the importaNT features
    """
    worst_ones = []
    for i in range(0, (X_frame.shape[1])):
        if i not in indexes:
            worst_ones.append(i)

    indexes = worst_ones
    print(worst_ones)
    return indexes


@celery.task(name='auto_feature_selection')
def auto_feature_selection(file_path):

    df = pd.read_csv(file_path, sep=',|;', header=0, engine='python')
    df.fillna(0)

    fraction_to_keep = 0.6

    k = int(fraction_to_keep * df.shape[1])

    hyperopt_trained_feature_space = [{'model': 'SelectKBest',
                                       'params':
                                       {
                                           'score_func': 'f_regression',
                                           'k': k
                                       }},
                                      {'model': 'SelectKBest',
                                       'params':
                                       {
                                           'score_func': 'mutual_info_regression',
                                           'k': k
                                       }}
                                      ]

    # print(hyperopt_trained_outlier_space[0]['model'])
    # print(hyperopt_trained_outlier_space[0]['param']['min_samples'])
    # print('')

    voting_out_percents = [0.0] * df.shape[1]

    for i in hyperopt_trained_feature_space:
        indexes = feature_selection(file_path, i)
        for j in indexes:
            voting_out_percents[j] += 1

    # print(len(hyperopt_trained_feature_space))

    for i in voting_out_percents:
        i = ((i) * 1.0) / (1.0 * (len(hyperopt_trained_feature_space)))

        # print(voting_out_percents)

    return voting_out_percents


@celery.task(name='model_training_classification')
def model_training_classification(file_path, classification_model_space):
    """
    pipeline for classification 

    Arguments:
    file_path -- path for file to apply pipeline
    classification_model_space --  provide space to search

    Returns:
    cost -- mean validation_score 
    score --  mean test_score
    y_predicted -- mean y_predicted

    """

    print("==============================================")
    print("model_training_classification_started...")

    print(classification_model_space)

    if (type(file_path) is not 'pandas.core.frame.DataFrame'):
        df = pd.read_csv(file_path, sep=',|;', header=0, engine='python')
    else:
        df = file_path

    Y = df['label']
    X = df.drop(['label'], axis=1)

    if classification_model_space['model'] == "BernoulliNB":
        clf = BernoulliNB(**classification_model_space['param'])

    elif classification_model_space['model'] == "DecisionTreeClassifier":
        clf = sklearn.tree.DecisionTreeClassifier(
            **classification_model_space['param'])

    elif classification_model_space['model'] == 'RandomForestClassifier':
        clf = sklearn.ensemble.RandomForestClassifier(
            **classification_model_space['param'])

    elif classification_model_space['model'] == "KNeighborsClassifier":
        clf = sklearn.neighbors.KNeighborsClassifier(
            **classification_model_space['param'])

    elif classification_model_space['model'] == "GaussianProcessClassifier":
        clf = sklearn.gaussian_process.GaussianProcessClassifier(
            **classification_model_space['param'])

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0)

    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)

    cost = 1 - accuracy_score(y_test, y_test_pred)

    train_sizes, train_scores, test_scores = learning_curve(clf, X, Y, cv=3,
                                                            verbose=3,
                                                            n_jobs=-1)
    test_score = np.mean(test_scores)

    print("model_training_classification_ended...")
    print("==============================================\n")

    return cost, test_score, y_test_pred


@celery.task(name='auto_model_training_classification')
def auto_model_training_classification(
    file_path,
    voting_algo_percent=0.5,
    model_space=[
        {'model': 'DecisionTreeClassifier',
         'param': {'criterion': 'gini', 'max_features': 'log2', 'presort': True}},
        {'model': 'BernoulliNB',
         'param': {'alpha': 0.8}}
    ]
):
    """
    auto pipeline for classification 

    Arguments:
    file_path -- path for file to apply pipeline
    model_space --  optional-> can provide spaces for searching

    Returns:
    cost -- mean validation_score 
    score --  mean test_score
    y_predicted -- mean y_predicted

    """
    total_y_pred = 0
    mean_cost = 0
    mean_test_score = 0

    for i in model_space:
        cost, test_score, y_predicted = model_training_classification(
            file_path, i)
        total_y_pred = y_predicted + total_y_pred
        mean_cost += cost
        mean_test_score += test_score

    total_y_pred = total_y_pred >= voting_algo_percent * len(model_space)
    y_predicted = total_y_pred.astype('int')

    mean_cost = mean_cost / len(model_space)
    mean_test_score = mean_test_score / len(model_space)

    return mean_cost, mean_test_score, y_predicted


@celery.task(name='pipeline_classification')
def pipeline_classification(file_path, space):
    """
    pipeline for classification 

    Arguments:
    file_path -- path for file to apply pipeline
    space --  provide space to search

    Returns:
    cost -- mean validation_score 
    score --  mean test_score
    y_predicted -- mean y_predicted

    """
    space_missing_values_treatment = space['space_missing_values_treatment']
    space_outlier_treatment = space['space_outlier_treatment']
    space_feature_selection = space['space_feature_selection']
    space_model_training = space['space_model_training']

    if space_missing_values_treatment['model'] == 'auto':
        _, file_path = auto_treat_missing_values(file_path)
    else:
        _, file_path = treat_missing_values(
            file_path, space_missing_values_treatment)

    if space_outlier_treatment['model'] == 'auto':
        voting_percents, file_path = auto_detect_outliers(file_path)
    else:
        y_predicted, file_path = detect_outliers(
            file_path, space_outlier_treatment)

    if space_feature_selection['model'] == 'auto':
        idx, file_path = auto_feature_selection(file_path)
    else:
        indexes, file_path = feature_selection(
            file_path, space_feature_selection)

    if space_model_training['model'] == 'auto':
        cost, score, y_predicted = auto_model_training_classification(
            file_path)
    else:
        cost, score, y_predicted = model_training_classification(
            file_path, space_model_training)

    return cost, score, y_predicted


@celery.task(name='model_training_regression')
def model_training_regression(file_path, regression_model_space):
    """
    pipeline for regression 

    Arguments:
    file_path -- path for file to apply pipeline
    regression_model_space --  provide space to search

    Returns:
    validation_score -- mean validation_score 
    test_score --  mean test_score
    y_predicted -- mean y_predicted

    """

    if (type(file_path) is not 'pandas.core.frame.DataFrame'):
        df = pd.read_csv(file_path, sep=',|;', header=0, engine='python')
    else:
        df = file_path

    Y = df['label']
    X = df.drop(['label'], axis=1)

    print("==================================================")
    print("Regression  Started...")
    print(regression_model_space)

    if regression_model_space['model'] == "LinearRegression":
        reg = LinearRegression(fit_intercept=regression_model_space['param'][
            'fit_intercept'], normalize=regression_model_space['param']['normalize'])

    elif regression_model_space['model'] == "Lasso":
        reg = Lasso(fit_intercept=regression_model_space['param']['fit_intercept'],
                    normalize=regression_model_space['param']['normalize'],
                    alpha=regression_model_space['param']['alpha'],
                    positive=regression_model_space['param']['positive']
                    )

    elif regression_model_space['model'] == "Ridge":
        reg = Ridge(fit_intercept=regression_model_space['param']['fit_intercept'],
                    normalize=regression_model_space['param']['normalize'],
                    alpha=regression_model_space['param']['alpha'],
                    solver=regression_model_space['param']['solver']
                    )

    elif regression_model_space['model'] == "ElasticNet":
        reg = ElasticNet(fit_intercept=regression_model_space['param']['fit_intercept'],
                         l1_ratio=regression_model_space['param']['l1_ratio'],
                         normalize=regression_model_space[
                             'param']['normalize'],
                         alpha=regression_model_space['param']['alpha']
                         )

    elif regression_model_space['model'] == "LogisticRegression":
        reg = LogisticRegression(penalty=regression_model_space['param']['penality'],
                                 C=regression_model_space['param']['C'],
                                 class_weight=regression_model_space['param'][
            'class_weights'],
            solver=regression_model_space[
                                     'param']['solver'],
            fit_intercept=regression_model_space['param']['fit_intercept'])

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0)
    reg.fit(x_train, y_train)

    y_pred = reg.predict(x_test)

    train_sizes, train_scores, valid_scores = learning_curve(
        reg, X, Y, train_sizes=[50, 80, 110], cv=5, scoring="r2")

    test_score = np.average(train_scores)
    validation_score = np.average(valid_scores)

    print("regression_scores:" + str(validation_score))

    print("regression done...")
    print("==================================================\n")

    return validation_score, test_score, y_pred


@celery.task(name='auto_model_training_regression')
def auto_model_training_regression(file_path, model_space=[
    {'model': 'ElasticNet',
     'param': {'normalize': 'True', 'alpha': 0.44739075200501477, 'fit_intercept': 'True', 'precompute': 'False', 'l1_ratio': 0.0419633465417355}},

    {'model': 'Ridge',
     'param': {'normalize': 'True', 'alpha': 0.5892942905571601, 'fit_intercept': 'True', 'solver': 'cholesky'}},

    {'model': 'ElasticNet',
     'param': {'normalize': 'True', 'alpha': 0.9395251021487494, 'fit_intercept': 'True', 'precompute': 'False', 'l1_ratio': 0.9723562142631096}}
]):
    """
    Auto-pipeline for regression 

    Arguments:
    file_path -- path for file to apply pipeline
    model_space -- optional -> can provide different spaces to search too

    Returns:
    cost -- mean cost  
    score --  mean score
    y_predicted -- mean y_predicted

    """
    mean_test_score = 0
    mean_validation_score = 0
    mean_y_pred = 0
    for i in model_space:
        validation_score, test_score, y_pred = model_training_regression(
            file_path, i)
        mean_test_score += test_score
        mean_validation_score += validation_score
        mean_y_pred += y_pred

    mean_validation_score = mean_validation_score / len(model_space)
    mean_test_score = mean_test_score / len(model_space)
    mean_y_pred = mean_y_pred / len(model_space)

    return mean_validation_score, mean_test_score, mean_y_pred


@celery.task(name='pipeline_regression')
def pipeline_regression(file_path, space):
    """
    pipeline for regression

    Arguments:
    file_path -- path for file to apply pipeline
    space -- arguments for different pipeline ,use 'auto' if predefined algos are used

    Returns:
    cost -- mean cost  
    score --  
    y_predicted --

    """
    space_missing_values_treatment = space['space_missing_values_treatment']
    space_outlier_treatment = space['space_outlier_treatment']
    space_feature_selection = space['space_feature_selection']
    space_model_training = space['space_model_training']

    if space_missing_values_treatment['model'] == 'auto':
        _, file_path = auto_treat_missing_values(file_path)
    else:
        _, file_path = treat_missing_values(
            file_path, space_missing_values_treatment)

    if space_outlier_treatment['model'] == 'auto':
        voting_percents, file_path = auto_detect_outliers(file_path)
    else:
        y_predicted, file_path = detect_outliers(
            file_path, space_outlier_treatment)

    if space_feature_selection['model'] == 'auto':
        idx, file_path = auto_feature_selection(file_path)
    else:
        indexes, file_path = feature_selection(
            file_path, space_feature_selection)

    if space_model_training['model'] == 'auto':
        cost, score, y_predicted = auto_model_training_regression(file_path)
    else:
        cost, score, y_predicted = model_training_regression(
            file_path, space_model_training)

    return cost, score, y_predicted


@celery.task(name='pipeline')
def pipeline(file_path, space, problem_type='auto'):
    """
        pipeline for classification and regression

        Arguments:
        file_path -- path for file to apply pipeline
        space -- arguments for different pipeline ,use 'auto' if predefined algos are used
        problem_type -- 'auto' : if automatic 

        Returns:
        cost -- mean cost  
        score --  
        y_predicted --

    """

    if (problem_type == 'auto'):
        df = pd.read_csv(file_path, sep=',|;', header=0, engine='python')

        if len(df['label'].unique()) < 0.25 * df.shape[0]:
            cost, score, y_predicted = pipeline_classification(
                file_path, space)
        else:
            cost, score, y_predicted = pipeline_regression(file_path, space)

    elif (problem_type == 'classification' or problem_type == 0):
        cost, score, y_predicted = pipeline_classification(file_path, space)
    elif (problem_type == 'regression' or problem_type == 1):
        cost, score, y_predicted = pipeline_regression(file_path, space)
    else:
        raise Exception(
            'Wrong argument passed in problem_type\n needed ,"auto",0 or 1')

    return cost, score, y_predicted
