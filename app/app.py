import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Insurance Predictor")

st.title("Прогноз стоимости медицинской страховки")
st.write("Данное приложение использует модель машинного обучения (XGBoost) для оценки ежегодных медицинских расходов клиента")

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
model = joblib.load(model_path)

st.sidebar.header("Данные клиента")
age = st.sidebar.slider("Возраст", 18, 100, 30)
bmi = st.sidebar.number_input("Индекс массы тела (BMI)", 10.0, 60.0, 25.0)
children = st.sidebar.selectbox("Количество детей", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.radio("Курильщик?", ["yes", "no"])
sex = st.sidebar.selectbox("Пол", ["male", "female"])
region = st.sidebar.selectbox("Регион", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Рассчитать стоимость"):
    input_df = pd.DataFrame([{
        'age': age, 'sex': sex, 'bmi': bmi,
        'children': children, 'smoker': smoker, 'region': region
    }])
    
    prediction = model.predict(input_df)[0]
    
    st.subheader(f"Прогноз расходов: ${prediction:,.2f}")
    if smoker == 'yes' and bmi > 30:
        st.error("Внимание: комбинация курения и высокого BMI значительно увеличивает стоимость")
    elif prediction < 5000:
        st.success("Низкая категория риска")