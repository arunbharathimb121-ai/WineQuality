from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as ras
from preprocess import prep_data,file_data
from imblearn.over_sampling import SMOTE

import pickle
import joblib
from pathlib import Path

def train_model(X_train,X_test,y_train,y_test):
    models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForestClassifier": RandomForestClassifier(),
    
        "SVC": SVC(C=1.5, kernel='rbf', probability=True, random_state=42), 
        "KNeighboursClassification": KNeighborsClassifier(n_neighbors=7, weights='distance'),
        "DecisionTreeClassification": DecisionTreeClassifier(max_depth=4, random_state=42),
        "XGBClassifier": XGBClassifier(
            n_estimators=500,
            learning_rate=0.1,     
            max_depth=6,            
            random_state=42
        )
    }
    smote=SMOTE()
    X_train,y_train=smote.fit_resample(X_train,y_train)
    xgb_ratio=(y_train==0).sum()/(y_train==1).sum()
    models["XGBClassifier"].set_params(scale_pos_weight=xgb_ratio)
    best_model = None
    best_accuracy = 0
    for name, model in models.items():
        y_proba=model.fit(X_train, y_train).predict_proba(X_test)
        points=ras(y_test,y_proba[:,-1])
        print(name, f"{points:.2f}")
        if points>best_accuracy:
            best_accuracy=points
            best_model=model
    print(f"\nPeak Accuracy is {best_accuracy:.2f} from model {best_model}.")
    return best_model

def save_model(model,scaler):
    model_dir=Path("/mnt/mydata/AI-ASSIST")
    model_dir.mkdir(exist_ok=True,parents=True)
    model_path=model_dir/"model.pkl"
    scaler_path=model_dir/"scaler.pkl"
    encoder_path=model_dir/"label_encoder.pkl" 
    jpath=model_dir/"model.joblib" 
    with open("model.pkl","wb") as f:
        pickle.dump(model,f)

    print("Successfully stored model")
    joblib.dump(model,jpath)

    with open("scaler.pkl","wb") as f:
        pickle.dump(scaler,f)
    print("Successfully stored scaler")


if __name__=="__main__":
    print("Start Training Pipeline ......................")

    path="/mnt/mydata/AI-ASSIST/WineQT.csv"
    df=file_data(path)

    X_train,X_test,y_train,y_test,scaler=prep_data(df)
    m=train_model(X_train,X_test,y_train,y_test)
    save_model(m,scaler)
    print("Training completed succesfully.........!!!!!!!!!!!!!")
    

