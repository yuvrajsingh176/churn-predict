import streamlit as st
import pandas as pd
import numpy as np      
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

#load the trained model
model=tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

## streamlit app

st.title('Customer churn prediction')
st.write('This app predicts whether a customer will churn or not based on their information.')
st.write('Please enter the following information:')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age', 18,  92)
balance=st.number_input('Balance')
creditscore=st.number_input('Credit Score')
numofproducts=st.number_input('Number of Products', 1, 4)
isactive=st.selectbox('Is Active', [1, 0])
estimatedsalary=st.number_input('Estimated Salary')
tenure=st.number_input('Tenure',0,10)
has_creditcard=st.selectbox('Has Credit Card', [0, 1])

# prepare the input data for prediction
input_data = pd.DataFrame({ 
    'CreditScore': [creditscore],   
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [numofproducts],
    'HasCrCard': [has_creditcard],
    'IsActiveMember': [isactive],
    'EstimatedSalary': [estimatedsalary]
})  

#one hot encoding for geography
geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#combine the onehot encoded to input data   
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)


#scale the input data

input_data_scaled = scaler.transform(input_data)


#prediction
prediction = model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

if prediction_proba>0.5:
    st.write("Customer is likely to exit the bank")    
else:
      st.write("Customer is likely to stay with the bank")

      st.write(f"Probability of exiting: {prediction_proba:.2f}")
st.write(f"Probability of staying: {1-prediction_proba:.2f}")