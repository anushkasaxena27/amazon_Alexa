from mmap import PAGESIZE
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
st.set_page_config(page_title = 'NLP for Amazon Alexa Reviews',layout="wide",page_icon=':smile:',initial_sidebar_state='collapsed')
st.title("Sentiment Analysis of Amazon Alexa Reviews")

def get_sentiment(feedback):
    if feedback == 1:
        return "Positive🤩"
    if feedback == 0:
        return "Negative😡"

def load_data():
    data = pd.read_csv("amazon_alexa.tsv", sep="\t",parse_dates = ['date'],dayfirst = True)
    data.sort_values(by = 'date',inplace = True)
    data['sentiment'] = data['feedback'].apply(get_sentiment)
    return data


df = load_data()
if st.sidebar.checkbox("Show raw data"):
    df.drop(columns = ['variation','feedback'],inplace = True)
    st.write(df)

if st.sidebar.checkbox("Visualize"):
    st.header("Tweet sentiment Count")
    counter = df.sentiment.value_counts().reset_index()
    c1, c2 = st.columns([1,3])
    c1.write(counter)
    fig = px.pie(counter,'index','sentiment')
    c2.plotly_chart(fig, use_container_width=True)
    
