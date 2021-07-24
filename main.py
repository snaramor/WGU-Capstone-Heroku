#
# WGU C964 Capstone Project
# Equipment Faults in Manufacturing Environments
# Sean Naramor
# July 23, 2021
#
####################################
# This file performs the data manipulation and runs the machine learning algorithm
####################################
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import streamlit as st
from PIL import Image
import numpy as np
import os


def main():
    path = os.path.dirname(__file__) + '/'
    from sklearn.ensemble import RandomForestClassifier
    banner = Image.open(path + 'banner.jpg')
    st.image(banner)
    st.write("""
    # Equipment Fault Prediction
    This application demonstrates how a machine learning predictive algorithm can be used to detect future failure 
    events in manufacturing equipment.
    """)
    # This data has been filtered down to a single process we can measure and parsed into a readable format
    df = pd.read_csv(path + 'data_parsed.csv')
    print(f'Raw data shape: {pd.array(df.iloc[:, [0, 1, 2, 3, 4, 5, 6]]).shape}')
    print(f'Replacing {len(df.query("intensity == 0"))} zeroes inside column "intensity"')
    df['intensity'].replace(0, np.NaN, inplace=True)
    df['intensity'].fillna(int(df['intensity'].median()), inplace=True)
    print(f'Replacing {len(df.query("temp3 == 0"))} zeroes inside column "temp3"')
    df['temp3'].replace(0, np.NaN, inplace=True)
    df['temp3'].fillna(df['temp3'].median(), inplace=True)
    st.subheader('The Data Set:')
    st.dataframe(df)
    st.write('This data set contains 18-months worth of sensor data from a tool used in semi-conductor manufacturing.')
    st.write('Initially, this existed as over 500,000 data points spread out among over 6,300 files, similar '
             'to the image below.')
    rawData = Image.open(path + 'raw_data.png')
    st.image(rawData)
    st.write('After obfuscating sensitive data, filtering, and other data preparation tasks the final '
             'amount of data points sits at just over 2000')
    st.markdown('<hr>', unsafe_allow_html=True)

    st.subheader('Data Attributes: ')
    st.write(df.describe())
    st.markdown('<hr>', unsafe_allow_html=True)

    st.subheader('Data Graphs: ')
    st.write("Recorded Equipment Parameters (01/01/2020 - 06/30/2021):")
    fig = px.line(df.iloc[:, [0, 1, 2, 3, 4, 5]], )
    st.plotly_chart(fig)
    st.write('Hint: Double click on graph to reset the view')

    st.subheader('Outcome Graph:')
    fig2 = px.line(df.iloc[:, 6], line_shape='linear', range_y=[0, 1.5])
    st.plotly_chart(fig2)
    st.write('Hint: Double click on graph to reset the view')
    st.markdown('<hr>', unsafe_allow_html=True)

    st.subheader('Feature Correlation Matrix: ')
    corr = df.corr()
    corr_heatmap, corr_heatmap_ax = plt.subplots()
    sns.set(font_scale=0.75)
    sns.heatmap(corr, ax=corr_heatmap_ax, annot=True)
    st.write(corr_heatmap)
    st.markdown('<hr>', unsafe_allow_html=True)

    st.subheader('PCA Decomposition')
    # Our Input Parameters (tool sensors)
    X = df.iloc[:, [0, 1, 2, 3, 4, 5]].values
    # Our Output (Equipment fault: yes or no)
    Y = df.iloc[:, 6].values
    # split our data to create a set to train with and a set to test against
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
    print(f'Shape of Initial Training Dataset: {pd.array(X_train, dtype="float").shape}')
    print(f'Shape of Initial Testing Dataset: {pd.array(X_test, dtype="float").shape}')
    pca = PCA()
    pca.fit_transform(X_train)
    pca.transform(X_test)
    variance, vx = plt.subplots()
    print(f'PCA Variance: {pca.explained_variance_ratio_}')
    variance_df = pd.DataFrame({'Variance': pca.explained_variance_ratio_,
                                'Params': ['temp1', 'temp2', 'intensity', 'warm', 'temp3', 'gas']})
    sns.barplot(x='Params', y='Variance', data=variance_df, color='orange')
    st.write(variance)
    print('PCA found that columns 3, 4, and 5 are irrelevant to outcome, dropping from table')
    # Decomposing our data reveals that the 'warm', 'temp3', and 'gas' columns are inconsequential and can be removed
    # So we redefine our X and re-create our train/test split
    X = df.iloc[:, [0, 1, 2]].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
    print(f'Shape of Revised Training Dataset: {pd.array(X_train, dtype="float").shape}')
    print(f'Shape of Revised Testing Dataset: {pd.array(X_test, dtype="float").shape}')
    st.write('PCA Results showed us that only 2 or 3 columns of data were relevant to the outcome.')
    st.write('In the final model, the "warm", "temp3", and "gas" columns were removed from the dataset')
    st.markdown('<hr>', unsafe_allow_html=True)

    st.subheader('Model Accuracy:')
    # Create our ML model structure
    classifier = RandomForestClassifier()
    # Train our model with our training data (12 months' worth)
    print('Running Random Forest Classification algorithm on training data...')
    classifier.fit(X_train, Y_train)
    print('Algorithm Complete')
    print('Testing model on test data...')
    Y_pred = classifier.predict(X_test)
    # Test our model against our test data and show the user the results
    acc = round(accuracy_score(Y_test, Y_pred) * 100, 2)
    print(f'Test Complete. Accuracy score was {acc}%')
    st.write(f'Accuracy using the Random Forest Classification model: {acc}%')
    st.write("Confirmation via Confusion Matrix:")
    sns.set(font_scale=1.5)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    heat, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='g', ax=ax)
    st.write(heat)
    st.markdown('<hr>', unsafe_allow_html=True)

    def get_input():
        temp1 = st.sidebar.slider('temp1', 0, 46, 29)
        temp2 = st.sidebar.slider('temp2', 800, 1000, 874)
        intensity = st.sidebar.slider('intensity', 0, 6286, 3452)
        user_dict = {
            'temp1': temp1,
            'temp2': temp2,
            'intensity': intensity
        }
        features = pd.DataFrame(user_dict, index=[0])
        return features

    user_input = get_input()
    prediction = classifier.predict(user_input)
    st.subheader('User Input: ')
    st.write('Using the sliders to the left, adjust the parameters to determine if the parameters will cause a '
             'fault in the equipment.')
    if prediction[0] == 1:
        st.write('Fault: Yes')
    else:
        st.write('Fault: No')
