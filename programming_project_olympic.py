import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# creating the olympics dataframe 
olympic_df = pd.read_csv('https://raw.githubusercontent.com/MaH1996SdN/programming_project/master/athlete_events.csv')
# creating regions dataframe
regions_df = pd.read_csv('https://raw.githubusercontent.com/MaH1996SdN/programming_project/master/noc_regions.csv')

# adding some explanation about the data
st.header('120 years of Olympic history')
st.subheader('basic bio data on athletes and medal results from Athens 1896 to Rio 2016')
st.write('This is a historical dataset on the modern Olympic Games, including all the Games from Athens 1896 to Rio 2016.')

# adding a checkbox for displayin raw data
show_raw_data = st.checkbox('Show raw data')
if show_raw_data:
    st.subheader('Raw data')
    st.write(olympic_df)
