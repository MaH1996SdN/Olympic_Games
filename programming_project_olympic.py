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


#******************************INTRODUCTION*************************************************************************************************************************
# adding some explanation about the data
st.header('120 years of Olympic history')
st.write('basic bio data on athletes and medal results from Athens 1896 to Rio 2016')
st.write('This is a historical dataset on the modern Olympic Games, including all the Games from Athens 1896 to Rio 2016. The Winter and Summer Games were held in the same year up until 1992. After that, they staggered them such that Winter Games occur on a four year cycle starting with 1994, then Summer in 1996, then Winter in 1998, and so on')

st.text("")
st.text("")


# adding a checkbox for displayin raw data
show_raw_data = st.checkbox('Show raw data')
if show_raw_data:
    st.subheader('Raw data')
    st.write(olympic_df)


#********************************ORGANAZING CLEANING DATA************************************************************************************************************

# droping extra columns: id, games, event in olympic dataframe
olympic_df.drop(['ID', 'Games','Event'], axis=1, inplace=True)

# dropping extra column: note in regions dataframe
regions_df.drop('notes', axis=1, inplace=True)

# merging 2 dataframes in order to find countries 
olympic_df = pd.merge(olympic_df, regions_df, on='NOC', how='left')

# renaming region to "Country" for better understanding
olympic_df.rename(columns = {'region':'Country'}, inplace = True)

# dropping extra column in olympic dataframe: Team/ includes unrelated data
olympic_df.drop('Team', axis=1, inplace=True)

# removing rows containing null values of Age column 
olympic_df.dropna(subset=['Age'], inplace=True)

# filling null values of Height column with median value
olympic_df['Height'].fillna(olympic_df['Height'].mean(), inplace=True)

# filling null values of Weight column with median value
olympic_df['Weight'].fillna(olympic_df['Weight'].mean(), inplace=True)

# filling null values with "0"
olympic_df['Medal'].fillna(0, inplace=True)

# replacing string values of Medals with integer values
olympic_df.Medal.replace({'Gold':1, 'Silver':2, 'Bronze':3}, inplace=True)

# removing rows containing null countries
olympic_df = olympic_df[olympic_df['Country'].notna()]

# removing the rows containing "Art Competitions in the "Sport" column
olympic_df.drop(olympic_df.loc[olympic_df['Sport']=='Art Competitions'].index, inplace=True)

Medals = olympic_df.loc[olympic_df['Medal'] != 0 ]

femaleParticipants = olympic_df.loc[(olympic_df['Sex'] == 'F')]
femaleMedalists = femaleParticipants.loc[femaleParticipants['Medal'] != 0]



# adding a checkbox for displaying cleaned data
show_cleaned_data = st.checkbox('Show cleaned data')
if show_cleaned_data:
    st.subheader('Cleaned data')
    st.write(olympic_df)

st.text("")
st.text("")


#**************************EXPLORATION********************************************************************************************************************
st.subheader('Exploration')
st.text("")

with st.expander('Exploration on countries'):
    col_1, col_2 = st.columns(2)
    with col_1:
        st.write('Participants by country/ Top 5')
        # top 5 countries by number of participants
        st.dataframe(olympic_df.Country.value_counts().head(5))
        st.text("")
    with col_2:
        st.write('Participants by country/ Last 5')
        # last 5 countries by number of participants
        st.dataframe(olympic_df.Country.value_counts().tail(5))
        st.text("")

    col_3, col_4 = st.columns(2)
    with col_3:
        st.write('Number of medals by country/ Top 5')
        # top 5 countries by number of medals
        st.dataframe(Medals.Country.value_counts().head(5))
        st.text("")
    with col_4:
        st.write('Number of medals by country/ Last 5')
        # last 5 countries by number of medals
        st.dataframe(Medals.Country.value_counts().tail(5))
        st.text("")

with st.expander('Exploration on sports'):
    col_5, col_6 = st.columns(2)
    with col_5:
        st.write('Top 5 sports by number of medals')
        # top 5 sports by number of medals
        st.dataframe(Medals.Sport.value_counts().head(5))
        st.text("")
    with col_6:
        st.write('Last 5 sports by number of medals')
        # last 5 sports by number of medals
        st.dataframe(Medals.Sport.value_counts().tail(5))
        st.text("")

with st.expander('Exploration on women in Olympics'):
    col_3, col_4 = st.columns(2)
    with col_3:
        st.write('female participants by country')
        # number of female participants by country
        st.dataframe(femaleParticipants.Country.value_counts().head(5))
        st.text("")
    with col_4:
        st.write('Top 5 countries with female medalists')
        # number of female participants by country
        femaleMedalists = femaleParticipants.loc[femaleParticipants['Medal'] != 0]
        st.dataframe(femaleMedalists.Country.value_counts().head(5))

