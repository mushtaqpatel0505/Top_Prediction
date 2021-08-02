#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import lightgbm as lgb

#Initialize the flask App
app = Flask(__name__)
model = lgb.Booster(model_file='lgb_regressorr.txt')

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
    df = pd.DataFrame(final_features, columns = ['s','t'])
    df['s'] = np.log(df['s'])
    df['t'] = np.log(df['t'])
    prediction = model.predict(df, num_iteration=model.best_iteration)
    if prediction[0]>0.99:
        output = 0.99
    else:
        output = prediction[0]

    return render_template('index.html', prediction_text='Prediction of top is : {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
