import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import category_encoders as ce
import plotly.express as px

st.set_page_config(page_title="🐧 Penguin Classifier", layout="wide")
st.title("🐧 Penguin Classifier - Обучение и предсказание")
st.write("👨‍🏫 Работа с датасетом пингвинов")

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

st.subheader("📋 Случайные 10 строк")
st.dataframe(df.sample(10), use_container_width=True)

st.subheader("📊 Визуализация данных")
col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(df, x="species", color="island", barmode="group", title="Распределение видов по островам")
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = px.scatter(df, x="bill_length_mm", y="flipper_length_mm", color="species", title="Длина клюва vs Длина крыла")
    st.plotly_chart(fig2, use_container_width=True)
