#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import lightgbm as lgb

#Initialize the flask App
app = Flask(__name__)
model = lgb.Booster(model_file='lgb_regressor.txt')

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features, num_iteration=model.best_iteration)

    output = prediction[0]

    return render_template('index.html', prediction_text='Prediction of top is : {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)