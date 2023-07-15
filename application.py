from flask import Flask, request, render_template, jsonify
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle

application = Flask(__name__)
app= application

# import ridge and scaler pickle-
standard_scaler= pickle.load(open("models/scaler.pkl", "rb"))
ridge_model= pickle.load(open("models/ridge.pkl", "rb"))

# create homepage-
@app.route("/")
def index():
    return render_template("index.html")

# function for prediction-
@app.route("/predictdata", methods= ["GET", "POST"])
def predict_datapoint():
    if request.method=="POST":

        # first we will retrieve all the values inputed by us in the form and store it in respective variables
        Temperature= float(request.form.get("Temperature"))
        RH= float(request.form.get("RH"))
        Ws= float(request.form.get("Ws"))
        Rain= float(request.form.get("Rain"))
        FFMC= float(request.form.get("FFMC"))
        DMC= float(request.form.get("DMC"))
        ISI= float(request.form.get("ISI"))
        Classes= float(request.form.get("Classes"))
        region= float(request.form.get("region"))

        # we scale all these new data points using scaler pickle-
        new_data_scaled= standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, region]])

        # we predict the result i.e. FWI using the ridge pickle-
        result= ridge_model.predict(new_data_scaled)

        # displaying the result in the home.html page itself-
        return render_template("home.html", results= result[0])


    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")
