import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

pipeline = joblib.load('iris_model_pipeline.pkl')

iris = load_iris()
target_names = iris.target_names

st.title('Iris Flower Species Prediction')
st.markdown('Enter the measurements below to predict the Iris flower species.')

sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.4)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.4)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.3)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

if st.button('Predict'):
    
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

   
    prediction_proba = pipeline.predict_proba(input_data)
    predicted_class_index = np.argmax(prediction_proba, axis=1)[0]
    predicted_class_name = target_names[predicted_class_index]
    
    st.subheader('Prediction Results')
    st.success(f'The predicted species is: **{predicted_class_name}**')
    
    st.write('Prediction Probabilities:')
    for i, prob in enumerate(prediction_proba[0]):
        st.write(f'- **{target_names[i]}**: {prob:.2f}')
