import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.title("Build A Beautiful Machine Learning Web App With Streamlit And Scikit-learn")
st.write("## Explore different Classifier")
name_dataset=st.sidebar.selectbox("Select the datset" , ('Breast cancer' ,'Iris','WineQuality' ,"Diabetes"))
st.write("#### Selecteed datset : " ,name_dataset)
name_classifier= st.sidebar.selectbox("Select the Classifier name" , ("KNN" ,"SVM","Random Forest"))
st.write("#### Selected classifier : " , name_classifier)

def get_dataset(name_dataset):
    if name_dataset == "Iris":
        data= datasets.load_iris()

    elif name_dataset == 'Breast cancer':
        data= datasets.load_breast_cancer()

    elif name_dataset == 'Diabetes':
        data= datasets.load_diabetes()
    else :
        data = datasets.load_wine()

    x=data.data
    y=data.target
    return x, y

x,y=get_dataset(name_dataset)
st.write("Shape of the dataset : " ,x.shape )


def add_paramter(name_classifier):
    para=dict()
    if name_classifier=="SVM":
        C= st.sidebar.slider("C",0.1,10.0)
        para["C"] =C
    elif name_classifier=="KNN":
        K= st.sidebar.slider("K",1,15)
        para["K"] = K
    else:
        max_depth=st.sidebar.slider("max_depth",2 ,7)
        number_estimator=st.sidebar.slider("number of estimator",1 ,100)
        para["max_depth"] = max_depth
        para["number_estimator"] = number_estimator
    return para

para=add_paramter(name_classifier)

def get_classifier(name_classifier, para):
    classify = None
    if name_classifier == 'SVM':
        classify = SVC(C=para['C'])
    elif name_classifier == 'KNN':
        classify = KNeighborsClassifier(n_neighbors=para['K'])
    else:
        classify = classify = RandomForestClassifier(n_estimators=para['number_estimator'], 
            max_depth=para['max_depth'], random_state=1234)
    return classify

classify = get_classifier(name_classifier, para)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.6 , random_state=10)
classify.fit(x_train,y_train)
y_predict= classify.predict(x_test)
accuaracy=accuracy_score(y_test,y_predict)
st.write("Accuracy = " ,accuaracy)

# FOR PLOTING
pca=PCA(2)
X_projected = pca.fit_transform(x)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

# x3 = X_projected[:, 2]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.colorbar()

st.pyplot(fig)