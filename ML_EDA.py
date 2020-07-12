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
import xgboost
from sklearn.preprocessing import LabelEncoder, StandardScaler,Normalizer
from sklearn.neighbors import KNeighborsClassifier

import xgboost

def main():
    st.title("Streamlit Auto ML")

    activities = ["Regression", "Classification"]
    choice = st.sidebar.selectbox("Choose the activity", activities)

    if choice == "Regression":
        st.subheader("Automated Regression Model")

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

            if st.checkbox("Correlation"):
                sns.heatmap(df.corr(), annot=True)
                st.pyplot()

            if st.checkbox("Missing values Count"):
                ms = df.isnull().sum()
                st.text(ms)

            if st.checkbox("Handle the missing values"):
                for cols in df:
                    if df[cols].dtype == 'O':
                        df[cols] = df[cols].fillna(df[cols].mode()[0])


                    else:
                        df[cols] = df[cols].fillna(df[cols].mean())

                st.dataframe(df.head())
                st.success("Your missing values are now gone")

            if st.checkbox("Columns"):
                st.text(df.columns.to_list())

            columnns = df.columns.to_list()
            scolumns = st.multiselect("Select columns",columnns)

            if scolumns:
                    st.dataframe(df[scolumns])

            cols = df.columns.to_list()


            # cp = st.multiselect("Select two columns for count plot", cols)
            # if len(cp) == 2:
            #
            #     sns.countplot(df)
            #     st.pyplot()
            #
            # else:
            #     st.error("Select only two columns")


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
                        st.dataframe(df.head())
                        st.success("Given categorical column is converted to numerical")

                    else:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col])
                        st.dataframe(df)
                        st.success("Given categorical column is converted to numerical")


            cols = df.columns.to_list()
            cp = st.multiselect("Select two columns for scatter plot", cols)
            if len(cp) ==2:

                plt.scatter(df[cp[0]], df[cp[1]], c="g")
                plt.plot()
                st.pyplot()

            else:
                st.error("Select only two columns")

            if st.checkbox("Pairplot"):
                sns.pairplot(df)
                st.pyplot()

            st.markdown("""
            <b>Model Building</b>, 
            """
            , unsafe_allow_html=True)


            for cols in df:
                if df[cols].dtype == 'O':
                    df[cols] = df[cols].fillna(df[cols].mode()[0])


                else:
                    df[cols] = df[cols].fillna(df[cols].mean())

            for col in cat_cols:
                if df[col].nunique() > 2:
                    dummy = pd.get_dummies(df[col], drop_first=True)
                    df.drop(col, axis=1, inplace=True)
                    df = df.join(dummy)

                else:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])

            col = df.columns.to_list()
            clmn = st.selectbox("Select the output column of your model", col)
            y = df.loc[:, clmn]
            y = pd.DataFrame(y)
            X = df.drop([clmn], axis=1)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            mx = StandardScaler()
            X_train = mx.fit_transform(X_train)
            y_train = mx.fit_transform(y_train)
            X_test = mx.transform(X_test)

            if st.checkbox("Dist Plot"):
                sns.distplot(X_train)
                st.pyplot()







        else:
            st.error("Please upload the files")




    if choice == "Classification":
        st.subheader("Classification")

        data = st.file_uploader("Upload your files", type="csv")

        if data is not None:
            df = pd.read_csv(data)



            columns = df.columns.to_list()





if __name__ == '__main__':
    main()


