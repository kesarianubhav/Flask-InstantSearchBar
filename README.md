
To run the app type => 
  FLASK_APP=run.py FLASK_DEBUG=True flask run 


Dependencies used =>
=> Python 3 ( python 2 will give an error since I am using the function yield )
=> Pandas (Just for loading the csv , I could have done it with the traditional open function also , but then it would have become too slow . Other than that there is no use of pandas in the logic or the app.)
=> Flask ( and its extension flask-wtforms  because wtforms are much more handy and secure forms than the typical html forms . )

