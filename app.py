import pandas as pd
import pickle
import numpy as np
import sklearn
from flask import Flask,render_template , request,jsonify



app = Flask(__name__)

@app.route("/")
def Home():
    return render_template('index.html')

@app.route('/predict' ,methods = ['POST','GET'])
def results():
    gre_score = float(request.form['gre score'])
    toefl_score= float(request.form['toefl score'])
    university_rating = float(request.form['university rating'])
    sop = float(request.form['sop'])
    lor = float(request.form['lor'])
    cgpa =float(request.form['cgpa'])
    research = float(request.form['research'])

    X = np.array([[gre_score ,toefl_score,university_rating,sop,lor,cgpa,research]])

    scaled_model = pickle.load(open('scaler.pkl','rb'))
    X_std = scaled_model.transform(X)
    

   
    model = pickle.load(open('Admission.Predictor_model.pkl','rb'))

    Y_prediction = model.predict(X_std)
    return jsonify({'Model Prediction': float(Y_prediction)})



if __name__ == "__main__":
    app.run(debug=True,port = 9457)