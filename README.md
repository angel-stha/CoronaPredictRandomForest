## ML-RandomForestRegressor
A simple project using MultiOutputRegressor to predict new corona cases and new death cases on basis of stringency index, No of hospitals_bed per thousand, HDI, Handwashing facilities in any region 

### Prerequisites
All prerequisites are mentioned in requirements.text

### Project Structure
This project has four major parts :
1. model.py - This contains code for our Machine Learning model i.e MultiOutputRegression on the corona.csv dataset provided on "https://ourworldindata.org/coronavirus-source-data"
2. app.py - This contains Flask APIs that receives the features from index.html and predict the cases using model saved and displays  it.
3. templates - This folder contains the HTML template to allow user to enter features for corona cases prediction and visualize the cases occurred giving the date and country in plot.

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -

2. Run app.py using below command to start Flask API
```
python app.py
```
or
```
flask run
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

4. Enter the necessary details and predict.
