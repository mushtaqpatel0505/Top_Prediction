#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
#import pickle
#import lightgbm as lgb
import pandas as pd
import catboost as cb

#Initialize the flask App
app = Flask(__name__)
cb_model = cb.CatBoostRegressor(loss_function='RMSE')
cb_model.load_model('model_name')
#model = lgb.Booster(model_file='lgb_regressorr.txt')
"""
#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')
"""
#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    """
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    df = pd.DataFrame(final_features, columns = ['s','t'])
    df['s'] = np.log(df['s'])
    df['t'] = np.log(df['t'])
    #prediction = model.predict(df, num_iteration=model.best_iteration)
    """
 
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    df = pd.DataFrame.from_dict(data)

    # predictions
 

    df['s'] = np.log(df['s'])
    df['t'] = np.log(df['t'])


    prediction = cb_model.predict(df)
    if prediction[0]>0.99:
        output = 0.99
    else:
        output = prediction[0]
        # send back to browser
    outputs = {'top':output}
    # return data
    return jsonify(results=outputs)
    #return render_template('index.html', prediction_text='Prediction of top is : {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
