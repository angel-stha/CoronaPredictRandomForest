import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
df= pd.read_csv("covid.csv").fillna(0.0)
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]


    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)


    output = round(prediction[0][0], 2)
    output2 = (prediction[0][1])


    return render_template('index.html', prediction_text='Total cases of corona on basis of HDI is {output} and stringency index is {output2}'.format(output=output, output2=output2))
@app.route('/plot_countrycases',methods=['POST'])
def plot_countrycases():

    features = [data for data in request.form.values()]
    location = features[0]
    start_date = features[1]
    target =features[2]
    y1 = df[(df['location'] == location) & (df['date'] > start_date)][[target]]
    x1 = df[(df['location'] == location ) & (df['date'] > start_date)][['date']]
    ax1 = plt.plot(range(0,len(x1)), y1, 'bo--')
    plt.title(target + " from " + start_date + " in " + location)
    plt.xlabel("Date from " + start_date)
    plt.ylabel(target + " of " + location +"from " + start_date)
    plt.savefig("./static/location.jpg")

    return render_template('index.html', plot_text="location.jpg")


if __name__ == "__main__":
    app.run(debug=True)