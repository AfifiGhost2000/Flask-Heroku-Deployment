import os
from flask import Flask, request, flash, redirect, send_file, render_template
import pandas as pd
import numpy as np
import pickle 
from livereload import Server
from model import get_label_encode


app = Flask(__name__)
app.debug = True

model = pickle.load(open('model.pkl','rb'))


# Set up the main route
@app.route('/', methods=["GET", "POST"])
def main():

    if request.method == "POST":
        # Extract the input from the form
        #customer_id = request.form.get("Number")
        #city = request.form.get("City")
        #gender = request.form.get("Gender")
        #age = request.form.get("Age")


        # Create DataFrame based on input
        #input_variables = pd.DataFrame([[customer_id, city, gender, age]],
        #                            columns=['Number','City', 'Gender', 'Age'],
        #                           dtype=float,
        #                            index=['input'])

        # Get the model's prediction
        # Given that the prediction is stored in an array we simply extract by indexing
        #prediction = model.predict(input_variables)[0]

        int_features = [get_label_encode(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)

        output = round(prediction[0], 2)

    
    
        # We now pass on the input from the from and the prediction to the index page
        return render_template("index.html", result = '$ {}'.format(output))
                              
                                    
    # If the request method is GET
    return render_template("index.html")

if __name__ == '__main__':
    app.run(port=5000)
    server = Server(app.wsgi_app)
    server.serve()