from preprocess import prep_data,file_data
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,roc_curve,RocCurveDisplay
import shap
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


def load_model(path):
    with open(path,"rb") as f:
        model=pickle.load(f)
    return model

def eval_model(model,X_test,y_test):
    y_proba=model.predict_proba(X_test)
    y_pred=model.predict(X_test)
    ypp=model.predict_proba(X_test)[:,1]
    cm=confusion_matrix(y_test,y_pred)

    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    k=roc_auc_score(y_test,ypp)
    fpr,tpr,thresholds=roc_curve(y_test,ypp)
    RocCurveDisplay(fpr=fpr,tpr=tpr).plot()
    plt.title("ROC CURVE")
    plt.show()

def model_explain(model,X_test):
    ex=shap.TreeExplainer(model)
    sv=ex.shap_values(X_test)
    shap.summary_plot(sv[:,:,1],X_test,plot_type="bar")
    plt.title("SHAP Analysis(Feature influencing model)")

    force_plot=shap.force_plot(ex.expected_value[1],sv[:, :, 1][0],X_test[0])
    shap.save_html("save.html",force_plot)

if __name__=="__main__":
    X_train,X_test,y_train,y_test,scaler=prep_data(file_data('WineQT.csv'))
    model=load_model('model.pkl')
    eval_model(model,X_test,y_test)
    model_explain(model,X_test)