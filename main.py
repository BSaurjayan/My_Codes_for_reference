import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

@st.cache
def get_data(filename):
    taxi_data = pd.read_csv(filename)
    return taxi_data

with header:
    st.title("Welcome to my awesome Demo!!!")
    st.text("This is the NYC taxi dataset, in this demo we look into the tansactions of the taxis in NYC....")

    

with dataset:
    st.header("NYC taxi dataset")
    st.text("Found the dataset in Kaggle")

    
    taxi_data = get_data("data\yellow_tripdata_2021-01.csv")

    st.subheader('Pickup location ID distribution: ')
    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)






with features:
    st.header("The features I created")
    





with model_training:
    st.header('Time to train the model:')
    st.text("We choose hyperparameters of the model to find the optimum performance of model.")

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider("Max_depth of the model:", min_value =10, max_value=100, value=20)
    n_estimators = sel_col.selectbox("Number of trees : ", options = [100, 200,300,400, 'No Limit'], index=0)

    sel_col.text("List of input feature in data: ")
    sel_col.write(taxi_data.columns)

    input_feature = sel_col.text_input("Which feature as input:", 'trip_distance')

    if n_estimators == 'No Limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    

    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]

    regr.fit(X,y)
    prediction = regr.predict(y)


    disp_col.subheader("Mean absolute error of the model:")
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader("Mean squared error of the model:")
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader("R2 score of the model:")
    disp_col.write(r2_score(y,prediction))




