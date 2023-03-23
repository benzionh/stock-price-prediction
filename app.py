import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

AAPL = "AAPL Historical Data.csv"
AMZN = "AMZN Historical Data.csv"
FB = "FB Historical Data.csv"
GOOG = "GOOG Historical Data.csv"
GOOGL = "GOOGL Historical Data.csv"
MSFT = "MSFT.csv"
NFLX = "NFLX Historical Data.csv"

st.title('Your tool for stock trading')
user_input = st.text_input('Enter MSFT.csv, AAPL Historical Data.csv, AMZN Historical Data.csv, FB Historical Data.csv, GOOG Historical Data.csv, GOOGL Historical Data.csv, NFLX Historical Data.csv : ')
df = pd.read_csv(user_input)
df = df.set_index(pd.DatetimeIndex(df['Date'].values))

# Describing data
st.write(df.describe())

# Visualization
st.subheader('Closing price Vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

# Create a new dataframe with close column
data = df.filter(['Close'])                                                                 
#convert dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.8)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


# load model
model = load_model('keras_model.h5')

test_data=scaled_data[training_data_len-60:,:]
#create x_test and y_test
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])


#Convert data into numpy array
x_test=np.array(x_test)
#Reshape the data
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#Getting the model's predicted values
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)






# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions']=predictions
#visualize the data
st.subheader("Predictions Vs Original")
fig1 = plt.figure(figsize=(16,8))
plt.title('Predictions Vs Original')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price ($)',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','val','Predictions'],loc='lower right')
st.pyplot(fig1)

new_df = pd.read_csv(user_input)
new_df = new_df.filter(['Close'])
last_60_days = new_df[-60:].values
# Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
# create an empty list
X_test = []
# append the past 60 days
X_test.append(last_60_days_scaled)
# convert the X_test dataset to a numpy
X_test = np.array(X_test)
# reshape the data
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
# get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
st.write("The predicted price is ")
st.write(pred_price)
