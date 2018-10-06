from flask import Flask

from utils import random_string

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
t_lastName = t3
t_Name = t4

import views
