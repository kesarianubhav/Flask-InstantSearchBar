import random
import string
import pickle
import pprint
from io import StringIO
import requests
import pandas as pd
import redis


class Store(object):

    def __init__(self, store_id, host='localhost', port=6379, db=0):

        self.pool = redis.ConnectionPool(host=host, port=port, db=db)
        self.r = redis.Redis(connection_pool=self.pool)
        self.key = store_id
        self.state = "STARTED"
        self.r.set(self.key, self.state)

    # @property
    # def state(self):
    #   result = r.get(self.key).decode('ascii')
    #     return result

    # @state.setter
    def setState(self, state):
        self.r.set(self.key, state)

    # @state.deleter
    def delState(self):
        del self.state

    def setNull(self):
        self.r.set(self.key, "NULL")

    def getState(self):
        r = self.r.get(self.key).decode()
        return r


def url_to_dataframe(url):

    r = requests.get(url, stream=True)
    r = r.content
    s = str(r, 'utf-8')
    data = StringIO(s)
    df = pd.read_csv(data)

    return df


def acha():
    print("DING DING")


def random_string():
    """
    Generate 16 Characters Random String
    Args: None
    Kwargs: None
    Returns:
        rs (str): random string
    Raises: None
    """
    rs = (''.join(random.choice(string.ascii_uppercase)
                  for i in range(16)))
    return rs


def storeData(class_instance, output_filename):

    try:
        output_file = output_filename + ".pkl"
        fptr = open(output_file, "wb")
        pickle.dump(class_instance, fptr)
        return output_file

    except Exception as e:
        print("Cant Pickle")
        return 0


def loadData(filename):

    try:
        instance = pickle.loads(filename)
        return instance

    except Exception as e:
        print("Cant Load File ")
        return 0
