import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import boto3
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# s3 = boto3.resource('s3')
# obj = s3.Object('angel-corona','owid-covid-data.csv' )
# body = obj.get()['Body'].read()
df= pd.read_csv("covid.csv").fillna(0.0)
x_stringex=df.iloc[:,[32,37,38,40]]
x_stringex.fillna(0.0)
y=df.iloc[:,[4,25]].fillna(0.0)
x_stringex_train, x_stringex_test,y1,y2 =train_test_split(x_stringex,y,test_size=0.2)
sc=StandardScaler()
x_string_train = sc.fit_transform(x_stringex_train)
x_string_test = sc.fit_transform(x_stringex_test)
Y1=sc.fit_transform(y1)
Y2=sc.fit_transform(y2)
model=RandomForestRegressor()
print(Y1.shape)
model.fit(x_string_train,y1)



pickle.dump(model,open('model.pkl','wb'))
