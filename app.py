import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from pmdarima import auto_arima

df = pd.read_csv("covid.csv").fillna(0.0)
app = Flask(__name__)
model = pickle.load(open('wrapper.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
        Compute new case and new death cases
        against the features through MultiOutputRegression.
        Parameters
        -------------
            stringency index(int): ranges from 0 to 100
                                   showing strict measure
                                   taken by government
            No of hospitals_bed per (int): indicates health facility
            handwashing_facility(int): prevention measure of covid
            HDI (int): Human Development Index
        Returns
        -------------
            output(int): predicted new_case
            output2(int): predicted new_death_case
    """
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0][0], 2)
    output2 = (prediction[0][1])
    return render_template('index.html',
                           prediction_text='Total new cases of '
                                           'corona predicted is {output} '
                                           'and new death cases '
                                           'is {output2}'
                           .format(output=output, output2=output2))


@app.route('/plot_countrycases', methods=['POST'])
def plot_countrycases():
    """
        plot the data of cases on basis of date of specific country
        Parameters
        -------------
            location : from form; user inputs location
            date : user inputs date starting from which they want to visualize
            target : selects from new covid cases or new death cases
        Returns
        -------------
            plot:  plot of data of cases on basis of date of specific country
    """
    features = [data for data in request.form.values()]
    location = features[0]
    start_date = features[1]
    target = features[2]
    y1 = df[(df['location'] == location) & (df['date'] > start_date)][[target]]
    x1 = df[(df['location'] == location) & (df['date'] > start_date)][['date']]
    plt.plot(range(0, len(x1)), y1, 'bo--')
    plt.title(target + " from " + start_date + " in " + location)
    plt.xlabel("Date from " + start_date)
    plt.ylabel(target + " of " + location + "from " + start_date)
    plt.savefig("./static/location.jpg")

    return render_template('index.html', plot_text="location.jpg")


@app.route('/forecast', methods=['POST'])
def forecast():
    """
        visulaize the difference between forcasted
        by ARIMA model and actual data.
        Parameters
        -------------
            location : from form; user inputs location
            date : user inputs date starting from which they want to visualize
            target : selects from new covid cases or new death cases
        Returns
        -------------
            plot: plot of ARIMA forecast
    """
    features = [data for data in request.form.values()]
    country_name = features[0]
    start_date = features[1]
    end_date = '2020-10-11'
    target = features[2]
    data_cp = df.loc[(df['location'] == country_name)]
    data_cp_train = data_cp.loc[(data_cp['date']) < start_date, target]
    data_cp_test = data_cp.loc[(data_cp['date']) > start_date]
    test_cp_date = data_cp_test.loc[(data_cp_test['date'] < end_date), target]
    # Set the range of parameters to use
    stepwise_model = auto_arima(data_cp[target], start_p=1, start_q=1,
                                max_p=30, max_q=30, start_P=0, seasonal=False,
                                d=2, trace=False, error_action='ignore',
                                stepwise=True)

    # Train and predict
    stepwise_model.fit(data_cp_train, start_ar_lags=2 * max(30, 30))
    forecast = stepwise_model.predict(n_periods=len(test_cp_date))

    # Plotting
    valid_num = len(test_cp_date)
    df_train = df.loc[(df['location'] == country_name), target]
    df_fcst = np.append(df_train[:-valid_num], forecast[:valid_num])
    dates = list(range(0, len(df_train)))
    plt.plot(dates, df_fcst)
    plt.plot(dates, df_train)
    plt.axvline(len(df_train) - valid_num - 1,
                linewidth=2, ls=':', color='grey', alpha=0.5)
    plt.title("Actual " + target + " vs predictions "
                                   "based on ARIMA for " + country_name)
    plt.legend(['Predicted cases', 'Actual cases',
                'Train-test split'], loc='upper left')
    plt.xlabel("Days from January 2020 to 11th October 2020")
    plt.ylabel(target)
    plt.savefig("./static/target.jpg")
    return render_template('index.html', forecast="target.jpg")


if __name__ == "__main__":
    app.run(debug=True)
