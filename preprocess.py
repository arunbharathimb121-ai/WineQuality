import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def file_data(path):
    data=pd.read_csv(path)
    return data

def scale_features(X,Xt):
    scaler=StandardScaler()
    Xf=scaler.fit_transform(X)
    Xe=scaler.transform(Xt)
    return Xf,Xe,scaler


def prep_data(df):
    df=file_data(df)
    X=df.drop(["Id","quality"],axis=1)
    
    y=df['quality'].apply(lambda x: 1 if x>=7 else 0)

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

    X_train,X_test,scaler=scale_features(X_train,X_test)

    
    return X_train,X_test,y_train,y_test,scaler

