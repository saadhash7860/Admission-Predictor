import pandas as pd
import pickle
import numpy as np
import sklearn
from flask import Flask,render_template , request,jsonify
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


app = Flask(__name__)

@app.route("/")
def Home():
    return render_template('index.html')

@app.route('/predict' ,methods = ['POST','GET'])
def results():
    GRE_Score = float(request.form['GRE_Score'])
    TOEFL_Score = float(request.form['TOEFL_Score'])
    University_Rating = float(request.form['University_Rating'])
    sop = float(request.form['sop'])
    lor = float(request.form['lor'])
    cgpa =float(request.form['cgpa'])
    research = float(request.form['research'])

    features = scaler.fit_transform([[GRE_Score,TOEFL_Score,University_Rating ,sop,lor,cgpa,research]])
    
    
   
    filename = 'Admission.Predictor_model.pickle'
    model = pickle.load(open(filename,'rb'))
    Y_prediction = model.predict(features)
    return jsonify({'Model Prediction': float(Y_prediction)})



if __name__ == "__main__":
    app.run(debug=True)