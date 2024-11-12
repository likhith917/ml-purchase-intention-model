import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler





from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier

##################
# To ignore warning #
##################
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
df = pd.read_csv("online_shoppers_intention.csv")

###################
# Feature Engineering #
###################


df['Weekend'] = df['Weekend'].replace((True, False), (1, 0))
df['Revenue'] = df['Revenue'].replace((True, False), (1, 0))



##############################
# Added Returning_Visitor column #
##############################
print("Step 4: Added Returning_Visitor column successfully")
condition = df['VisitorType']=='Returning_Visitor'
df['Returning_Visitor'] = np.where(condition, 1, 0)
df = df.drop(columns=['VisitorType'])

########################################
# Applying One Hot Encoding on Month column #
#########################################


ordinal_encoder = OrdinalEncoder()
df['Month'] = ordinal_encoder.fit_transform(df[['Month']])

####################################
# Checking correlation on Revenue column #
#####################################
print("Step 6: Checking correlation done successfully")
result = df[df.columns[1:]].corr()['Revenue']
result1 = result.sort_values(ascending=False)
      

X = df.drop(['Revenue'], axis=1)
y = df['Revenue']

####################################
# Preparing Train and Test Dataset#
####################################


X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.3, random_state = 0)



###############
# Model Pipeline #
###############
print("Step 9: model_pipeline function created done successfully")
def model_pipeline(X, model):
                              n_c = X.select_dtypes(exclude=['object']).columns.tolist()
                              c_c = X.select_dtypes(include=['object']).columns.tolist()

                              numeric_pipeline = Pipeline([
                              ('imputer', SimpleImputer(strategy='constant')),
                              ('scaler', MinMaxScaler())
                              ])

                              categorical_pipeline = Pipeline([
                              ('encoder', OneHotEncoder(handle_unknown='ignore'))
                              ])

                              preprocessor = ColumnTransformer([
                              ('numeric', numeric_pipeline, n_c),
                              ('categorical', categorical_pipeline, c_c)
                              ], remainder='passthrough')

                              final_steps = [
                              ('preprocessor', preprocessor),
                              ('smote', SMOTE(random_state=1)),
                              ('feature_selection', SelectKBest(score_func = chi2, k =
                              6)),
                              ('model', model)
                              ]
                              return IMBPipeline(steps = final_steps)
def select_model(X, y, pipeline=None):
                              classifiers = {}
                              c_d1 = {"DummyClassifier":
                              DummyClassifier(strategy='most_frequent')}
                              classifiers.update(c_d1)
                              c_d4 = {"RandomForestClassifier":
                              RandomForestClassifier()}
                              classifiers.update(c_d4)
                              c_d5 = {"DecisionTreeClassifier": DecisionTreeClassifier()}
                              classifiers.update(c_d5)
                              c_d9 = {"KNeighborsClassifier": KNeighborsClassifier()}
                              classifiers.update(c_d9)
                              c_d14 = {"SVC": SVC()}
                              classifiers.update(c_d14)
                              mlpc = {
                              "MLPClassifier (paper)":
                              MLPClassifier(hidden_layer_sizes=(27, 50),
                              max_iter=300,
                              activation='relu',
                              solver='adam',
                              random_state=1)
                              }
                              c_d16 = mlpc
                              classifiers.update(c_d16)
                              cols = ['model', 'run_time', 'roc_auc']
                              df_models = pd.DataFrame(columns = cols)
                              for key in classifiers:
                                                     start_time = time.time()
                                                     print()



                                                     pipeline = model_pipeline(X_train, classifiers[key])
                                                     cv = cross_val_score(pipeline, X, y, cv=10,
                                                     scoring='roc_auc')
                                                     row = {'model': key,

                                                    'run_time': format(round((time.time() -
                                                     start_time)/60,2)),
                                                    'roc_auc': cv.mean(),
                                                     }

                                                     df_models = pd.concat([df_models,
                                                     pd.DataFrame([row])], ignore_index=True)
                                                     df_models = df_models.sort_values(by='roc_auc',
                                                     ascending = False)
                              return df_models
models = select_model(X_train, y_train)

#############################
# Let’s see total model with score #
#############################

print(models)

###############################
# Accessing best model and training #
###############################

selected_model = MLPClassifier()
bundled_pipeline = model_pipeline(X_train, selected_model)
bundled_pipeline.fit(X_train, y_train)
y_pred = bundled_pipeline.predict(X_test)
print(y_pred)

###################
# ROC and AOC score #
###################
print("Step 16: ROC and AOC scores")
roc_auc = roc_auc_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)

print('ROC/AUC:', roc_auc)
print('Accuracy:', accuracy)
print('F1 score:', f1_score)

###################
# Classification report #
###################
print("Step 17: classification report generated successfully")
classif_report = classification_report(y_test, y_pred)
print(classif_report)
