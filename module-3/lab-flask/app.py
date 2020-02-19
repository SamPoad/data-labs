
# DEPENDENCIES
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

# create a Flask application instance, the name of the file you ran gets passed behind the scenes
app = Flask(__name__, static_url_path = '/static') 

# you could have global variables or global functions defined up here, or even SQLite database 

# Routes - how the server behaves when it receives a request (be it a GET or POST etc.) 
@app.route('/') # go to home directory with this path, localhost
def root(): # you never have to call the function, because the app.route will CALL the function
    return "Hello, world!"


@app.route('/page') # if the server gets a request to this, do the function below it
def render_page():
    return render_template(
        'page.html',
        name = 'world',
        img = 'myfig.png',
        ) # create an html template, and dynamically insert content into it,


# Create URL for prediction API and only allow POST requests
@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    model = joblib.load('./static/Models/GaussianNB.sav')
    data = request.get_json() # get data from POST request's parameters, 
    # and immediately convert/store as a JSON, basically a dictionary
    # print(data)
    df = pd.DataFrame(data) # so we take in the POST stuff, convert to a dataframe so we can then...
    pred = model.predict(df.values) # run a prediction off of our saved/loaded model
    return jsonify({'survived': pred[0]}) # return the prediction as a JSON to pass back to the user's browser


if __name__ == '__main__': # if this the file that's intentionally being executed, not a module, then do this thing
    app.run(port = 3000, debug = True) # run on port 3000 instead of flask's default 5000 and 
    # don't forget to set debug = False when going to production!

