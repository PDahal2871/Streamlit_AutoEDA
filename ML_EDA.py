import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("agg")
import seaborn as sns
import streamlit as st
import pandas_profiling
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

import xgboost

def main():
    st.title("Streamlit Auto ML")

    activities = ["EDA", "Model Building", "Plot"]
    choice = st.sidebar.selectbox("Choose the activity", activities)

    if choice == "EDA":
        st.subheader("Exploratory Data Analysis")

        data = st.file_uploader("Upload your files", type="csv")

        if data is not None:
            df = pd.read_csv(data)

            if st.checkbox("Head"):
                st.dataframe(df.head())

            if st.checkbox("Shape"):
                st.text(df.shape)

            if st.checkbox("Size"):
                st.text(df.size)

            if st.checkbox("Describe"):
                st.dataframe(df.describe())

            if st.checkbox("Missing values Count"):
                ms = df.isnull().sum()
                st.text(ms)

            if st.checkbox("Handle the missing values"):
                ms = df.isnull().sum()


            if st.checkbox("Columns"):
                st.text(df.columns.to_list())

            columnns = df.columns.to_list()
            scolumns = st.multiselect("Select columns",columnns)

            if scolumns:
                    st.dataframe(df[scolumns])

            vc = st.checkbox("Value Count of above selected Columns")

            if vc:
                if scolumns:
                    for elements in scolumns:
                        st.subheader(elements + " value count")
                        st.dataframe(df[elements].value_counts())
                else:
                    st.error("Please select a column")

            cat_cols =[]
            for col in columnns:
                if df[col].dtype == 'O':
                    cat_cols.append(col)

            if st.checkbox("Categorical columns"):
                if cat_cols is not None:
                    st.text(cat_cols)
                else:
                    st.error("No categorical columns")

            cat = st.multiselect("Select the categorical column you want to convert to numerical", cat_cols)
            if cat:
                for col in cat:
                    if df[col].nunique() > 2:
                        dummy = pd.get_dummies(df[col], drop_first=True)
                        df.drop(col, axis=1, inplace=True)
                        df = df.join(dummy)
                        st.dataframe(df)
                        st.success("Given categorical column is converted to numerical")

                    else:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col])
                        st.dataframe(df)
                        st.success("Given categorical column is converted to numerical")



        else:
            st.error("Please upload the files")




    if choice == "Model Building":
        st.subheader("Model Building")

        data = st.file_uploader("Upload your files", type="csv")

        if data is not None:
            st.text("YEss")

    if choice == "Plot":
        st.subheader("Plotting")

        data = st.file_uploader("Upload your files", type="csv")

        if data is not None:
            df = pd.read_csv(data)

            if st.checkbox("Correlation"):
                sns.heatmap(df.corr(), annot=True)
                st.pyplot()

            columns = df.columns.to_list()
            choice = st.multiselect("Select 2 columns for scatter plot", columns)
            try:
                if len(choice) <3 and len(choice) >0:
                    x = choice[0]
                    y = choice[1]
                    plt.scatter(df[x], df[y], c="green")
                    st.pyplot()

                else:
                    st.error("Please select two columns")

            except:
                st.error("Please select two columns")




if __name__ == '__main__':
    main()


