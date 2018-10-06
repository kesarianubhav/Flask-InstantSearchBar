from celery import Celery
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from utils import random_string
from utils import Store

from autocomplete_logic import Trie
from autocomplete_logic import loadData
from autocomplete_logic import populate_Trie

# Initialize flask app
UPLOAD_FOLDER = '../files'
ALLOWED_EXTENSIONS = {'csv'}

# Create a flask app
app = Flask(__name__)
app.secret_key = random_string()
app.config.from_object('settings')

df = loadData('data.csv')
(t1, t2, t3, t4) = populate_Trie(df)
t_firstName = t1
t_middleName = t2
t_surName = t3
t_Name = t4

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

import views
