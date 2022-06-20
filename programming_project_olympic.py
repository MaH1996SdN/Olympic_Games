# importing libraries
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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from PIL import Image


# creating the olympics dataframe 
olympic_df = pd.read_csv('https://raw.githubusercontent.com/MaH1996SdN/programming_project/master/athlete_events.csv')
# creating regions dataframe
regions_df = pd.read_csv('https://raw.githubusercontent.com/MaH1996SdN/programming_project/master/noc_regions.csv')


#******************************INTRODUCTION*************************************************************************************************************************

# adding some explanation about the data
st.header('120 years of Olympic history')
st.write('basic bio data on athletes and medal results from Athens 1896 to Rio 2016')
st.write('This is a historical dataset on the modern Olympic Games, including all the Games from Athens 1896 to Rio 2016. The Winter and Summer Games were held in the same year up until 1992. After that, they staggered them such that Winter Games occur on a four year cycle starting with 1994, then Summer in 1996, then Winter in 1998, and so on.')

st.text("")
st.text("")

image = Image.open('./2.jpg')
st.image(image)

st.text("")
st.text("")


show_raw_data = st.checkbox('Show raw data')
if show_raw_data:
    st.subheader('Raw data')
    st.write(olympic_df)



#********************************ORGANAZING AND CLEANING DATA************************************************************************************************************

# droping extra columns: id, games, event in olympic dataframe
olympic_df.drop(['ID', 'Games', 'City'], axis=1, inplace=True)

# dropping extra column: note in regions dataframe
regions_df.drop('notes', axis=1, inplace=True)

# merging 2 dataframes in order to find countries 
olympic_df = pd.merge(olympic_df, regions_df, on='NOC', how='left')

# renaming region to "Country" for better understanding
olympic_df.rename(columns = {'region':'Country'}, inplace = True)

# dropping extra column in olympic dataframe: Team/ includes unrelated data
olympic_df.drop(['Team', 'NOC'], axis=1, inplace=True)

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

# creating medals dataframe
Medals = olympic_df.loc[olympic_df['Medal'] != 0 ]

# creating gold medals dataframe
goldMedals = olympic_df.loc[olympic_df['Medal'] == 1]

# creating female participants dataframe
femaleParticipants = olympic_df.loc[(olympic_df['Sex'] == 'F')]

# creating female medalists dataframe
femaleMedalists = femaleParticipants.loc[femaleParticipants['Medal'] != 0]

# creating female gold medalists dataframe
femaleGoldMedalists = femaleParticipants.loc[femaleParticipants['Medal'] == 1]




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

# medals
with st.expander('Medals'):
   
   # creating 3 columns with srtreamlit to show data
    col_21, col_22, col_23 = st.columns(3)
    with col_21:
        st.metric('Number of Gold medals:', olympic_df['Medal'].value_counts()[1])
    with col_22:
        st.metric('Number of Silver medals:', olympic_df['Medal'].value_counts()[2])
    with col_23:
        st.metric('Number of Bronze medals:', olympic_df['Medal'].value_counts()[3])
    st.metric('Total number of medals:', Medals.shape[0])


# Countries
with st.expander('Exploration on countries'):
    col_1, col_2 = st.columns(2)
    with col_1:
        st.write('Number of participants by country/ Top 5')
        # top 5 countries by number of participants
        st.dataframe(olympic_df.Country.value_counts().rename(index='').head(5))
        st.text("")
    with col_2:
        st.write('Number of participants by country/ Last 5')
        # last 5 countries by number of participants
        st.dataframe(olympic_df.Country.value_counts().rename(index='').tail(5))
        st.text("")

    col_3, col_4 = st.columns(2)
    with col_3:
        st.write('Number of medals by country/ Top 5')
        # top 5 countries by number of medals
        st.dataframe(Medals.Country.value_counts().rename(index='').head(5))
        st.text("")
    with col_4:
        st.write('Number of medals by country/ Last 5')
        # last 5 countries by number of medals
        st.dataframe(Medals.Country.value_counts().rename(index='').tail(5))
        st.text("")

    col_5, col_6 = st.columns(2)
    with col_5:
        st.write('Number of gold medals by country/ Top 5')
        # top 5 countries by number of gold medals
        st.dataframe(goldMedals.Country.value_counts().rename(index='').head(5))
        st.text("")
    with col_6:
        st.write('Number of gold medals by country/ Last 5')
        # last 5 countries by number of gold medals
        st.dataframe(goldMedals.Country.value_counts().rename(index='').tail(5))
        st.text("")



# Sports
with st.expander('Exploration on sports'):
    col_9, col_10 = st.columns(2)
    with col_9:
        st.write('Top 5 sports by number of medals')
        # top 5 sports by number of medals
        st.dataframe(Medals.Sport.value_counts().rename(index='').head(5))
        st.text("")
    with col_10:
        st.write('Last 5 sports by number of medals')
        # last 5 sports by number of medals
        st.dataframe(Medals.Sport.value_counts().rename(index='').tail(5))
        st.text("")




# women in olympics
with st.expander('Exploration on women in Olympics'):
    col_13, col_14 = st.columns(2)
    with col_13:
        st.write('Number of female participants by country/ Top 5')
        # top 5 countries by number of female participants
        st.dataframe(femaleParticipants.Country.value_counts().rename(index='').head(5))
        st.text("")
    with col_14:
        st.write('Number of female participants by country/ Last 5')
        # last 5 countries by number of female participants
        st.dataframe(femaleParticipants.Country.value_counts().rename(index='').tail(5))
        st.text("")

    col_15, col_16 = st.columns(2)
    with col_15:
        st.write('Number of female medalists by country/ Top 5')
        # top 5 countries by number of medals by women
        st.dataframe(femaleMedalists.Country.value_counts().rename(index='').head(5))
        st.text("")
    with col_16:
        st.write('Number of female medalists by country/ Last 5')
        # last 5 countries by number of medals by women
        st.dataframe(femaleMedalists.Country.value_counts().rename(index='').tail(5))
        st.text("")

    col_17, col_18 = st.columns(2)
    with col_17:
        st.write('Number of female gold medalists by country/ Top 5')
        # top 5 countries by number of gold medals by women
        st.dataframe(femaleGoldMedalists.Country.value_counts().rename(index='').head(5))
        st.text("")
    with col_18:
        st.write('Number of female gold medalists by country/ Last 5')
        # last 5 countries by number of gold medals by women
        st.dataframe(femaleGoldMedalists.Country.value_counts().rename(index='').tail(5))
        st.text("")

# Describe
with st.expander('Describing dataframe'):
    st.dataframe(olympic_df.describe())

# Correlation
with st.expander('Correlation between features'):
    st.dataframe(olympic_df.corr())


# distrubution
with st.expander('Distrubutions'):
    col_19, col_20 = st.columns(2)
    with col_19:
        ax = plt.figure(figsize=(12,10))
        olympic_df.Age.hist()
        plt.title('Distrubution', fontweight='bold', fontsize=14)
        st.write(ax)
        st.caption('Distrubution of Participant\'s Age')
    with col_20:
        ax = plt.figure(figsize=(12,10))
        olympic_df.Height.hist()
        plt.title('Distrubution', fontweight='bold', fontsize=14)
        st.write(ax)
        st.caption('Distrubution of Participant\'s Height')

st.text("")
st.text("")

#**************************Charts********************************************************************************************************************

st.subheader('Charts')
st.text("")

# correlation graph
with st.expander('Heatmap'):
    ax = plt.figure(figsize=(12,10))
    sns.heatmap(olympic_df.corr(), annot=True)
    plt.title('Correlation', fontweight='bold', fontsize=14)
    st.write(ax)



# participants by sex
with st.expander('Percentage of participants'):
    ax = plt.figure(figsize=(10,6))
    labels = ['Male', 'Female']
    plt.pie(olympic_df['Sex'].value_counts(), labels=labels, autopct='%.2f%%', startangle=90, shadow=True)
    plt.title('Percentage of participants by sex', fontweight='bold', fontsize=14)
    plt.legend()
    st.write(ax)

# top 5 medalists
with st.expander('Top 5 medalists'):
    ax = plt.figure(figsize=(10,6))
    lb = Medals['Name'].value_counts().head(5)
    labels = lb.index
    plt.pie(Medals['Name'].value_counts().head(5),labels= labels, autopct='%.2f%%', startangle=90)
    plt.title('Top 5 medalists', fontweight='bold', fontsize=14)
    plt.legend()
    plt.legend(loc="upper left", prop={'size': 6})
    st.write(ax)

# top 10 countries with medals
with st.expander('Top 10 countries with medals'):
   total = Medals.Country.value_counts().reset_index(name='Medal').head(10)
   sns.catplot(x="index", y="Medal", data=total, height=6, kind="bar")
   plt.xlabel('Countries', fontsize=12 )
   plt.ylabel('Number of Medals', fontsize=12)
   plt.title('Top 10 countries by number of medals', fontweight='bold', fontsize=14)
   plt.xticks(rotation = 45)
   st.pyplot(plt.gcf())



# top 10 countries with female medalists
with st.expander('Top 10 countries with female medalists'):
    femaleMedalistByCountry = femaleMedalists.Country.value_counts().reset_index(name='Medal').head(10)
    sns.catplot(x='index', y='Medal', data=femaleMedalistByCountry, height=6, kind='bar')
    plt.xlabel('Countries' , fontsize=12 )
    plt.ylabel('Number of Medals', fontsize=12)
    plt.title('Top 10 countries with female medalists', fontweight='bold', fontsize=14)
    plt.xticks(rotation = 45)
    st.pyplot(plt.gcf())


# Age of medalists
with st.expander('Age of medalists'):
    ax = plt.figure(figsize=(10,6))
    plt.plot(Medals['Age'].value_counts().sort_index())
    plt.xticks(np.arange(5, 80, step=5))
    plt.title('Age of medalists', fontweight='bold', fontsize=14)
    plt.xlabel('Age', fontsize=12)
    st.write(ax)

# Height of medalists
with st.expander('Height of medalists'):
    ax = plt.figure(figsize=(10,6))
    plt.plot(Medals['Height'].value_counts().sort_index())
    plt.xticks(np.arange(130, 230, step=5))
    plt.title('Height of medalists', fontweight='bold', fontsize=14)
    plt.xlabel('Height', fontsize=12)
    st.write(ax)

# Number of Female and Male Athletes over time
with st.expander('Number of Female and Male Athletes over time'):
    maleParticipants = olympic_df.loc[(olympic_df['Sex'] == 'M')]
    f = femaleParticipants.groupby('Year')['Sex'].value_counts()
    m = maleParticipants.groupby('Year')['Sex'].value_counts()
    ax = plt.figure(figsize=(16, 14))
    plt.plot(f.loc[:,'F'], label = 'Female', color = 'red')
    plt.plot(m.loc[:,'M'], label = 'Male', color = 'blue')
    plt.title('Number of Female and Male Athletes over time', fontweight='bold', fontsize=14)
    plt.xlabel('Years', fontsize=12)
    plt.xticks(np.arange(1890, 2030, step=4))
    plt.xticks(rotation = 45)
    plt.legend()
    st.write(ax)

st.text("")
st.text("")

#**************************Modeling********************************************************************************************************************

# removing extra columns
olympic_df2 = olympic_df.drop(['Name', 'Season','Country'], axis=1)

# defining function for changing string values to numeric values for better modelling
def encode_df(dataframe):
    le = LabelEncoder()
    for column in dataframe.columns:
        dataframe[column] = le.fit_transform(dataframe[column])
    return dataframe

# changing string values to numeric values 
encode_df(olympic_df2)

st.subheader('Data Modeling')
st.text("")


with st.expander('modeling without KFold'):

        st.subheader('Predict medals')

        y_1 = olympic_df2.Medal

        select_model_1 = st.selectbox('Select model:', ['GaussianNB', 'RandomForestClassifier', 'DecisionTreeClassifier', 'KNeighborsClassifier'], key="firstselectbox")

        model_1 = GaussianNB()

        if select_model_1 == 'RandomForestClassifier':
            model_1 = RandomForestClassifier()
        elif select_model_1 == 'DecisionTreeClassifier':
            model_1 = DecisionTreeClassifier()
        elif select_model_1 == 'KNeighborsClassifier':
            model_1 = KNeighborsClassifier()

        

        test_size_1 = st.slider('Test size: ', min_value=0.1, max_value=0.9, step =0.1, key="firstslider")

        if st.button('RUN MODEL', key="firstbutton"):
            with st.spinner('Training...'):
                x_1 = olympic_df2.drop(['Medal'], axis=1)
                x_train, x_test, y_train, y_test = train_test_split(x_1, y_1, test_size=test_size_1, random_state=42)

                model_1.fit(x_train, y_train)

                y_pred = model_1.predict(x_test)

                accuracy_1 = accuracy_score(y_test, y_pred)

                st.write(f'Accuracy = {accuracy_1:.5f}')


with st.expander('modeling with KFold'):

        st.subheader('Predict medals')

        y_2 = olympic_df2.Medal

        select_model_2 = st.selectbox('Select model:', ['GaussianNB', 'RandomForestClassifier', 'DecisionTreeClassifier', 'KNeighborsClassifier'], key="secondselectbox")

        model_2 = GaussianNB()

        if select_model_2 == 'RandomForestClassifier':
            model_2 = RandomForestClassifier()
        elif select_model_2 == 'DecisionTreeClassifier':
            model_2 = DecisionTreeClassifier()
        elif select_model_2 == 'KNeighborsClassifier':
            model_2 = KNeighborsClassifier()

        
        x_2 = olympic_df2.drop(['Medal'], axis=1)
        test_size_2 = st.slider('Test size: ', min_value=0.1, max_value=0.9, step =0.1, key="secondslider")

        if st.button('RUN MODEL', key="secondbutton"):
            with st.spinner('Training...'):
                kf_two = KFold(n_splits=5, shuffle=True, random_state=42)
                accuracies = []
                i_2 = 0
                for train_index, test_index in kf_two.split(x_2):
                    i_2 += 1
                    model_2 = RandomForestClassifier(random_state=42)
                    x_train, x_test = x_2.iloc[train_index], x_2.iloc[test_index]
                    y_train, y_test = y_2.iloc[train_index], y_2.iloc[test_index]
                    model_2.fit(x_train, y_train)
                    y_pred = model_2.predict(x_test)
                    accuracy_2 = accuracy_score(y_pred, y_test)
                    accuracies.append(accuracy_2)
                    st.write(i_2, ') accuracy = ', accuracy_2)

                st.write(f'Mean accuracy: {np.array(accuracies).mean():.5f}')
          

#**************************Conclusion********************************************************************************************************************
st.text("")
st.text("")

# adding conclusion
st.subheader('Conclusion')
st.text("")
st.write("After using differenent classification models with and without KFold, it is concluded that the \"GaussionNB\" and \"KNeighborsClassifier\" have the highest accuracy score in comparison with \"RandomForestClassifier\" and \"DecisionTreeClassifier\". accuracy scores for the different models are listed below:")

col_30, col_31 = st.columns(2)
with col_30:
    st.markdown('**GaussionNB: 0.84**')
with col_31:
    st.markdown('**RandomForestClassifier: 0.83**')


col_32, col_33 = st.columns(2)
with col_32:
    st.markdown('**DecisionTreeClassifier: 0.75**')
with col_33:
    st.markdown('**KNeighborsClassifier: 0.84**')