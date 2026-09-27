import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

#loading model
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

st.set_page_config(page_title="Diabetes Dashboard", layout="wide")

#store last user input
if "last_input" not in st.session_state:
    st.session_state.last_input=None

#sidebar
st.sidebar.title("Diabetes ML App")
st.sidebar.info("Machine Learning based Diabetes Prediction System")

page = st.sidebar.radio("Navigate",["Dashboard","Predict","About"])

#Dashboard
if page == "Dashboard":
    st.title("Diabetes Prediction Dashboard")
    col1 , col2 , col3 = st.columns(3)

    col1.metric("Model Type" , "SVM")
    col2.metric("Features Used" , "8")
    col3.metric("status" , "Active")

    st.markdown("---") 

    st.subheader("Normal vs  Your Medical Data Comparison") 
    normal_values = [1,100,70,20,80,222,0.5,30]

    features = [
        "Pregnancies","Glucose","Blood Pressure","Skin Thickness","Insulin","BMI","Diabetes Pedigree Function","Age" 
    ]

    if st.session_state.last_input is not None:
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=features,
            y=normal_values,
            name="Normal Person"
        ))
        fig.add_trace(go.Bar(
                    x=features,
                    y=st.session_state.last_input,
                    name="Your input"
                ))

        fig.update_layout(
            barmode = "group",
            xaxis_title = "Medical Features",
            yaxis_title = "values",
            title = "Medical Features Comparison",
            template= "plotly_white",
            height = 500
        )

        st.plotly_chart(fig,use_container_width=True)

    else:
        st.info("Please make a prediction first to see comparison chart")  


# prediction  
elif page =="Predict":
    st.title("Predict Diabetes")
    col1,col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies",0,20)
        glucose = st.number_input("Glucose")
        blood_pressure =  st.number_input("Blood Pressure")
        skin_thickness =  st.number_input("Skin Thickness")

    with col2:
        insulin = st.number_input("Insulin")
        bmi = st.number_input("BMI")
        dpf=  st.number_input("Diabetes Pedigree Function")
        age =  st.number_input("Age")

    if st.button("Predict Now"):
        with st.spinner("Analyzing Medical Data...."):
            time.sleep(1.5)

            input_data = np.array([[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,dpf,age]])
            st.session_state.last_input = input_data[0]
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0][1]

        st.markdown("---") 

        if prediction == 1:
             st.error("Diabetic")

        else:
            st.success("Not Diabetic")   


        st.info(f"Probability of Diabetes: {round(probability * 100 , 2)}%")

        #Risk gauge meter
        gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Diabetes Risk %"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 40], "color": "green"},
                {"range": [40, 70], "color": "yellow"},
                {"range": [70, 100], "color": "red"},
            ],
        }
    )
)

        gauge.update_layout(height=400)
        st.plotly_chart(gauge, use_container_width = True)

#About 
else:
    st.title("About Project")
    st.write("""
    This project predicts whether a person is diabetic or not using a machine learning
      model trained on medical data.
    
      Model Used : Support Vector Machine(SVM)
      Dataset : PIMA Indian Diabetes Dataset 
      """ )       
