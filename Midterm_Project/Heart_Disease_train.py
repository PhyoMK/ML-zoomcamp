import pandas as pd
import numpy as np
import seaborn as sns
import pickle

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.metrics import roc_auc_score
# get_ipython().run_line_magic('matplotlib', 'inline')

#saving model

df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

#EDA 

df = df.rename(columns=
                {'resting bp s': 'resting blood pressure',
                'target': 'heartdisease'}
)

df.columns = df.columns.str.lower().str.replace(' ','_')
df.fillna(0)
df.st_slope.value_counts()

sex_values = {
    1: 'male',
    0: 'female',
}

df.sex = df.sex.map(sex_values)

chest_pain_type_values = {
    1: 'typical_angina',
    2: 'atypical_angina',
    3: 'non_anginal_pain',
    4: 'asymptomatic',
}

df.chest_pain_type = df.chest_pain_type.map(chest_pain_type_values)

fasting_blood_sugar_values = {
    1: 'true',
    0: 'false',
}

df.fasting_blood_sugar = df.fasting_blood_sugar.map(fasting_blood_sugar_values)

resting_ecg_values = {
    0: 'normal',
    1: 'st_t_wave_abnormal',
    2: 'lvh',
}

df.resting_ecg = df.resting_ecg.map(resting_ecg_values)

exercise_angina_values = {
    1: 'yes',
    0: 'no',
}

df.exercise_angina = df.exercise_angina.map(exercise_angina_values)

st_slope_values = {
    1: 'upsloping',
    2: 'flat',
    3: 'downsloping',
    0: 'flat',
}

df.st_slope = df.st_slope.map(st_slope_values)

#data categorization

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
numerical_columns   = list(df.dtypes[df.dtypes == 'int64'].index)
numerical_columns.append('oldpeak')

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ','_')

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.heartdisease.values
y_val = df_val.heartdisease.values
y_test = df_test.heartdisease.values

del df_train['heartdisease']
del df_val['heartdisease']
del df_test['heartdisease']

#data checking , mutual score and correlation

global_heartdisease_rate = df_full_train.heartdisease.mean()

# from IPython.display import display

numerical_columns.remove('heartdisease')

# for c in categorical_columns:
#     print(c)
#     df_group = df_full_train.groupby(c).heartdisease.agg(['mean', 'count'])
#     df_group['diff'] = df_group['mean']-global_heartdisease_rate
#     df_group['risk'] = df_group['mean']/global_heartdisease_rate
#     display(df_group)
#     print()
#     print()

from sklearn.metrics import mutual_info_score

def mutual_info_heartdisease_score(series):
    return mutual_info_score(series, df_full_train.heartdisease)

mi = df_full_train[categorical_columns].apply(mutual_info_heartdisease_score)

df_full_train[numerical_columns].corrwith(df_full_train.heartdisease)

train_dicts = df_train[categorical_columns + numerical_columns].to_dict(orient='records')
val_dicts = df_val[categorical_columns + numerical_columns].to_dict(orient='records')
test_dicts = df_test[categorical_columns + numerical_columns].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
x_train = dv.fit_transform(train_dicts)
x_val = dv.transform(val_dicts)
x_test = dv.transform(test_dicts)

#LOGISTICREGRESSION

model_log = LogisticRegression(C=1, max_iter=1000)
model_log.fit(x_train, y_train)

y_pred_log = model_log.predict_proba(x_val)[:,1]
auc_log = roc_auc_score(y_val, y_pred_log)

#Testing DecisionTree

# #Data tuning of decision
# scores = []
# for d in [4,5,6,7,10,15,20, None]:
#     for s in [1,2,5,10,15 ,20,100,200,500]:
#         dt = DecisionTreeClassifier(max_depth=d, min_samples_leaf=s)
#         dt.fit(x_train, y_train)

#         y_pred = dt.predict_proba(x_val)[:,1]
#         auc = roc_auc_score(y_val, y_pred)

#         scores.append((d,s,auc))
#         print('(%4s, %3d) --> %.3f' %(d, s, auc))

# columns = ['max_depth', 'min_samples_leaf', 'auc']
# df_scores = pd.DataFrame(scores, columns=columns)
# df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values=['auc'])
# sns.heatmap(df_scores_pivot, annot=True, fmt='.3f')

max_depth = 6
min_samples_leaf = 15
dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
dt.fit(x_train, y_train)

y_pred = dt.predict_proba(x_val)[:,1]
auc_decisiontree = roc_auc_score(y_val, y_pred)

#Randomforest

# #Data tuning of randomforest
# scores=[]

# for d in [5,10,15]:
#     for n in range(10,201,10):
#         rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=1)
#         rf.fit(x_train, y_train)
    
#         y_pred = rf.predict_proba(x_val)[:,1]
#         auc = roc_auc_score(y_val, y_pred)
#         scores.append((d, n, auc))

# df_scores = pd.DataFrame(scores, columns = ['max_depth', 'n_estimators', 'auc'])
# for d in [5,10,15]:
#     df_subset = df_scores[df_scores.max_depth == d]
#      plt.plot(df_subset.n_estimators, df_subset.auc, label='max_depth=%s'%d)
# plt.legend()

# max_depth = 10
# scores=[]

# for s in [1,2,3,5,10,50]:
#     for n in range(10,201,10):
#         rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth,min_samples_leaf = s, random_state=1)
#         rf.fit(x_train, y_train)
    
#         y_pred = rf.predict_proba(x_val)[:,1]
#         auc = roc_auc_score(y_val, y_pred)
#         scores.append((s, n, auc))

# df_scores = pd.DataFrame(scores, columns = ['min_samples_leaf', 'n_estimators', 'auc'])
# colors = ['black', 'green', 'blue', 'orange', 'red', 'grey']
# values = [1, 2, 3, 5, 10, 50]

# for s, col in zip(values, colors):
#     df_subset = df_scores[df_scores.min_samples_leaf == s]
#     plt.plot(df_subset.n_estimators, df_subset.auc, color=col, label='min_samples_leaf=%d'%s)
# plt.legend()

max_depth = 10
min_samples_leaf = 1
n_estimators = 50
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf = min_samples_leaf, random_state=1, n_jobs=-1 )
rf.fit(x_train, y_train)

y_pred = rf.predict_proba(x_val)[:,1]
auc_random_forest = roc_auc_score(y_val, y_pred)

#XGB 

features = dv.get_feature_names_out()
feature_names = list(features)
dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(x_val, label=y_val, feature_names=feature_names)
dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_names)

##XGB TUNING
# xgb_params= {
#     'eta':0.3,
#     'max_depth': 6,
#     'min_child_weight':1,
#     'objective': 'binary:logistic',
#     'nthread':8,
#     'seed':1,
#     'verbosity':1
# }

# model = xgb.train(xgb_params, dtrain, num_boost_round=10)
# y_pred = model.predict(dval)
# roc_auc_score(y_val, y_pred)

# def parse_xgb_output(output):
#     results = []

#     for line in output.stdout.strip().split('\n'):
#         num_iter, train_line, val_line = line.split('\t')

#         it = int(num_iter.strip('[]'))
#         train = float(train_line.split(':')[1])
#         val = float(val_line.split(':')[1])

#         results.append((it, train, val))
        
#     columns = ['num_iter', 'train_auc', 'val_auc']
#     df_results = pd.DataFrame(results, columns=columns)
#     return df_results

# watchlist = [(dtrain, 'train'), (dval, 'val')]
# scores = {}

# get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.4, \n    'max_depth': 12,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")

# key = 'min_child_weight=%s' %(xgb_params['min_child_weight'])
# scores[key] = parse_xgb_output(output)

# for min_child_weight, df_score in scores.items():
#     plt.plot(df_score.num_iter, df_score.val_auc, label=min_child_weight)

# plt.ylim(0.93,1)
# plt.legend()



#after tuning eta=0.4,max_depth = 12, min_child_weight=1


xgb_params= {
    'eta':0.4,
    'max_depth': 12,
    'min_child_weight':1,
    'objective': 'binary:logistic',
    'nthread':8,
    'seed':1,
    'verbosity':1
}

model_xgb = xgb.train(xgb_params, dtrain, num_boost_round=66)
y_pred = model_xgb.predict(dval)
auc_xgb = roc_auc_score(y_val, y_pred)

#Testing of model trainings

y_pred_log = model_log.predict_proba(x_test)[:,1]
auc_log = roc_auc_score(y_test, y_pred_log)

y_pred_tree = dt.predict_proba(x_test)[:,1]
auc_decisiontree = roc_auc_score(y_test, y_pred_tree)

y_pred_forest = rf.predict_proba(x_test)[:,1]
auc_random_forest = roc_auc_score(y_test, y_pred_forest)

y_pred_xgb = model_xgb.predict(dtest)
auc_xgb = roc_auc_score(y_test, y_pred_xgb)

# auc_log, auc_decisiontree, auc_random_forest, auc_xgb

#Saving model

model_file = 'model_rf.bin'
with open(model_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)

print(f'auc_random_forest {auc_random_forest}')
print(f'the model is save to {model_file}')




