from celery import Celery
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from utils import random_string
from utils import Store

# Initialize flask app
UPLOAD_FOLDER = '../files'
ALLOWED_EXTENSIONS = {'csv'}

# Create a flask app
app = Flask(__name__)
app.secret_key = random_string()
app.config.from_object('settings')


celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

import views
