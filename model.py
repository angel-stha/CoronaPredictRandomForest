import pandas as pd
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# s3 = boto3.resource('s3')
# obj = s3.Object('angel-corona','owid-covid-data.csv' )
# body = obj.get()['Body'].read()

# read data
df = pd.read_csv("covid.csv").fillna(0.0)
# feature selection
x = df.iloc[:, [25, 37, 38, 40]]
# output
Y = df.iloc[:, [5, 8]]
# train_test_split with train_size = 80% and test_size =20%
x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.2)
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# wrapper multioutputregression model with linear SVR
model = LinearSVR()
wrapper = MultiOutputRegressor(model)
wrapper.fit(X_train, y_train)

# save the model for prediction
pickle.dump(wrapper, open('wrapper.pkl', 'wb'))
