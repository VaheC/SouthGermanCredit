# Importing necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import boxcox
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pickle
import logging
import os
from woe_enc import WoeEncoder
import warnings
warnings.filterwarnings("ignore")

# Creating a logger function to log ML model building tasks
def create_log(log_file):
	'''Creates a logger

	   Parameters:
	   log_file: str, name of a log file to create
	   Result:
	   logger: a logger which outputs logging to a file and console
	'''
	if os.path.isfile(f'{log_file}.log'):
		os.remove(f'{log_file}.log')

	logging.basicConfig(level=logging.INFO, filemode='w')	 
	logger = logging.getLogger(__name__)

	c_handler = logging.StreamHandler()
	f_handler = logging.FileHandler(f'{log_file}.log')

	c_handler.setLevel(logging.INFO)
	f_handler.setLevel(logging.INFO)

	c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
								 datefmt='%d-%b-%y %H:%M:%S')
	f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
								 datefmt='%d-%b-%y %H:%M:%S')
	c_handler.setFormatter(c_format)
	f_handler.setFormatter(f_format)

	logger.addHandler(c_handler)
	logger.addHandler(f_handler)

	return logger

# Initiating a logger
logger = create_log('model_building')

# Connecting to Astra DB
# Please, take into account that it is possible to get an error 
# while connecting to the database. The problem is that the data 
# can be hybernated. So the issue can be solved by running 
# the code snippet below after 1-5 minutes.
try:
	logger.info('Connecting to database')

	cloud_config= {
	        'secure_connect_bundle': 'secure-connect-south-german-credit.zip'
	}
	auth_provider = PlainTextAuthProvider('utOxtpGgahrASICWfrGNZMGP', 
	                                      'zkSwAviwrSICsrE46+QORQzKRx.jZ,YZZUYfgPHs,jAnfOz0LMCI1tOpJ9ZIHn,NW0TjMebfC6kwIB3PUJ-_W2yG8CvOi0vWMHay+D-otMZE,_.wLzQq0SHw86LUwK6m')
	cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
	session = cluster.connect('credit_data')

	logger.info('Connection established')
except:
	logger.critical('Database connection failed', exc_info=True)
    print('The problem is that the data can be hybernated.')
    print('So the issue can be solved by running the code snippet below after 1-5 minutes')


# Getting data
logger.info('Loading the data')

data = pd.DataFrame(session.execute('SELECT * FROM "SouthGermanCredit";').all())
data = data.sort_values('id').drop(columns='id').reset_index(drop=True)

# Changing column names
logger.info('Data preprocessing has started.')

data.rename(columns={'laufkont': 'status',
                     'laufzeit': 'duration',
                     'moral': 'credit_history',
                     'verw': 'purpose',
                     'hoehe': 'amount',
                     'sparkont': 'savings',
                     'beszeit': 'employment_duration',
                     'rate': 'dti',
                     'famges': 'status_sex',
                     'buerge': 'other_debtors',
                     'wohnzeit': 'present_residence',
                     'verm': 'property',
                     'alter': 'age',
                     'weitkred': 'other_installment_plans',
                     'wohn': 'housing',
                     'bishkred': 'number_credits',
                     'beruf': 'job',
                     'pers': 'people_liable',
                     'telef': 'telephone',
                     'gastarb': 'foreign_worker',
                     'kredit': 'credit_risk'}, 
            inplace=True)

# Recoding credit_risk column
data['credit_risk'] = data['credit_risk'].map({1: 0, 0: 1})

# Creating hierarchical clustering feature
amt_scaler = StandardScaler()
data['sclog_amount'] = amt_scaler.fit_transform(np.log(1+data['amount']).values.reshape(-1, 1))

dur_scaler = StandardScaler()
data['sclog_duration'] = dur_scaler.fit_transform(np.log(1+data['duration']).values.reshape(-1, 1))

age_scaler = StandardScaler()
data['sclog_age'] = age_scaler.fit_transform(np.log(1+data['age']).values.reshape(-1, 1))

cluster_col_list = [ 'sclog_amount', 'sclog_duration', 'sclog_age']

mergings = linkage(data[cluster_col_list], method='ward') 
labels = fcluster(mergings, 25, criterion='distance')

data.drop(columns=['sclog_amount', 'sclog_duration', 'sclog_age'], inplace=True)
data['hclusters'] = labels

# Creating another feature using age and duration variables
data['dage'] = (100 - data['age']) / (12 * data['duration'])

# Cosntructing lists of continuous (numeric) and categorical features' names
num_col_list = ['age', 'amount', 'duration', 'dage']
cat_col_list = [col for col in data.columns 
                if col not in num_col_list and col!='credit_risk']
all_feat_list = num_col_list.copy()
all_feat_list.extend(cat_col_list)

# Applying Box-Cox transformation to numeric features
for col in num_col_list:
    data[col], temp_lmbda = boxcox(data[col], lmbda=None)

# Splitting the data into train and test sets
train_data, test_data = train_test_split(data, 
                                         test_size=0.2, 
                                         random_state=13,
                                         stratify=data['credit_risk'])

X_train = train_data[[i for i in train_data.columns if i!='credit_risk']].reset_index(drop=True)
y_train = train_data['credit_risk'].reset_index(drop=True).to_frame()

X_test = test_data[[i for i in test_data.columns if i!='credit_risk']].reset_index(drop=True)
y_test = test_data['credit_risk'].reset_index(drop=True).to_frame()

# Converting categorical features into onehot-encoded variables and splitting the data by applying train-test split
ohe_transformer = OneHotEncoder(drop='first', sparse=False)
ohe_transformer.fit(data[cat_col_list])

# Saving onehot transformer for deployment and continuing the data splitting
logger.info('Saving onehot transformer for deployment')

with open('ohe_transformer.pkl', 'wb') as file: 
    pickle.dump(ohe_transformer, file)

logger.info('Continuing the data preprocessing')

temp_data = pd.DataFrame(ohe_transformer.transform(data[cat_col_list]))
temp_data.columns = list(ohe_transformer.get_feature_names_out(cat_col_list))
data_ohe = pd.concat([data[num_col_list], temp_data], axis=1)
data_ohe['credit_risk'] = data['credit_risk'].values

train_data_ohe, test_data_ohe = train_test_split(data_ohe, 
                                                 test_size=0.2, 
                                                 random_state=13,
                                                 stratify=data_ohe['credit_risk'])

X_train_ohe = train_data_ohe[[i for i in train_data_ohe.columns if i!='credit_risk']].reset_index(drop=True)
y_train_ohe = train_data_ohe['credit_risk'].reset_index(drop=True).to_frame()

X_test_ohe = test_data_ohe[[i for i in test_data_ohe.columns if i!='credit_risk']].reset_index(drop=True)
y_test_ohe = test_data_ohe['credit_risk'].reset_index(drop=True).to_frame()

# Creating the first model: LogisticRegression with the onehot-encoded data + numeric features
logger.info('Creating the first model')

numeric_transformer = MinMaxScaler()
preprocessor = ColumnTransformer(transformers=[('num', 
                                                numeric_transformer, 
                                                num_col_list)],
                                 remainder='passthrough')
pipe_model = Pipeline([('preprocessor', preprocessor), 
                       ('model', LogisticRegression(C=0.05))])

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

cv_ohe = cross_validate(estimator=pipe_model,
                        X=X_train_ohe, 
                        y=y_train_ohe['credit_risk'],
                        cv=skf,
                        scoring='roc_auc', 
                        return_estimator=True)

logger.info(f"Average ROC-AUC for validation set is {np.round(cv_ohe['test_score'].mean(), 4)}.")

# Computing ROC-AUC for the test set
test_lr_ohe_ppred_list = []
for i_model in cv_ohe['estimator']:
    ppred_y_test = i_model.predict_proba(X_test_ohe)[:, 1]
    test_lr_ohe_ppred_list.append(ppred_y_test.reshape(-1, 1))

test_ppred_ohe = np.hstack(test_lr_ohe_ppred_list).mean(axis=1)
roc_test_ohe = roc_auc_score(y_test_ohe['credit_risk'], test_ppred_ohe)
roc_test_ohe = np.round(roc_test_ohe, 4)

logger.info(f"ROC-AUC of test set is {roc_test_ohe}.")

# Computing recall for the test set
fpr, tpr, thresholds = roc_curve(y_test, test_ppred_ohe)
opt_threshold = thresholds[np.argmax(tpr - fpr)]
recall_test_ohe = np.round(tpr[np.argmax(tpr-fpr)], 4)

logger.info(f"Recall of test set is {recall_test_ohe}.")

# Saving the model for deployment
logger.info('Saving the model for deployment')

with open('ohe_cv.pkl', 'wb') as file:
    pickle.dump(cv_ohe, file)

# Applying quantile binning to the numeric features
logger.info('Applying binning to numeric features')

kbd_dict = {}
for col in num_col_list:
    if col == 'age':
        bins = 5
    elif col == 'amount':
        bins = 9
    else:
        bins = 3

    kbd = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
    
    kbd.fit(X_train[col].to_frame())
    kbd_dict[col] = kbd
    X_train[f'{col}_bin'] = kbd.transform(X_train[col].to_frame())
    X_test[f'{col}_bin'] = kbd.transform(X_test[col].to_frame())

    cat_col_list.append(f'{col}_bin')

# Saving the binning strategy for deployment
logger.info('Saving the binning strategy for deployment')

with open('kbd_dict.pkl', 'wb') as file:
    pickle.dump(kbd_dict, file)

# Creating a function for iteraction features' generation

logger.info('Creating iteraction features')

def create_iter_feat(feat_list, X_train, y_train, X_test):
    '''Creates an iteraction feature out of specified features

       Params:
       feat_list: list, contains name of features for which 
                        an iteraction feature will be created
       X_train: dataframe, contains features for training set
       y_train: dataframe or series, contains a target variable for training set
       X_test: dataframe, contains features for test set

       Result:
       X_train: dataframe, the same X_train augmented with 
                           the iteraction feature
       X_test: dataframe, the same X_test augmented with 
                           the iteraction feature
       temp_train_map: dict, contains label mappings for the iteraction feature
       new_feat_list: list, contains names of created features 
    '''
    temp_iter_feat_name = '_'.join(feat_list) + '_iter'
    new_feat_list = []
    new_feat_list.append(temp_iter_feat_name)
    for col in feat_list:
        temp_col = col
        if temp_iter_feat_name not in X_train.columns:
            X_train[temp_iter_feat_name] = X_train[temp_col].astype(str)
            X_test[temp_iter_feat_name] = X_test[temp_col].astype(str)
        else:
            X_train[temp_iter_feat_name] = X_train[temp_iter_feat_name] + \
                                           '_' + X_train[temp_col].astype(str)
            X_test[temp_iter_feat_name] = X_test[temp_iter_feat_name] + \
                                          '_' + X_test[temp_col].astype(str)

    temp_train_list = list(X_train[temp_iter_feat_name].unique())
    temp_train_map = {temp_train_list[i]:i+1 for i in range(len(temp_train_list))}
    X_train[temp_iter_feat_name] = X_train[temp_iter_feat_name].map(temp_train_map)

    temp_test_list = list(X_test[temp_iter_feat_name].unique())
    temp_test_list = [i for i in temp_test_list if i not in temp_train_map]
    temp_test_map = {temp_test_list[i]:i+1+len(temp_train_list) for i in range(len(temp_test_list))}
    temp_train_map.update(temp_test_map)
    X_test[temp_iter_feat_name] = X_test[temp_iter_feat_name].map(temp_train_map)
    return X_train, X_test, temp_train_map, new_feat_list

# Generating iteraction features
iter_list = [['status', 'credit_history'],
             ['credit_history', 'savings'],
             ['credit_history', 'duration_bin'],
             ['savings', 'other_debtors'],
             ['savings', 'amount_bin'],
             ['duration_bin', 'other_debtors'],
             ['duration_bin', 'savings'],
             ['status', 'age_bin'],
             ['duration_bin', 'amount_bin'],
             ['duration_bin', 'age_bin'],
             ['age_bin', 'other_installment_plans'],
             ['age_bin', 'dage_bin']]

iter_map_dict = {}
for i_list in iter_list:
    X_train, X_test, temp_train_map, new_feat_list = create_iter_feat(feat_list=i_list, 
                                                                      X_train=X_train, 
                                                                      y_train=y_train, 
                                                                      X_test=X_test)
    iter_map_dict[iter_list.index(i_list)] = temp_train_map
    all_feat_list.extend(new_feat_list)
    cat_col_list.extend(new_feat_list)

# Saving the mapping dictionaries for the iteraction features to use during deployment
logger.info('Saving the mapping dictionaries for the iteraction features')

with open('iter_map_dict.pkl', 'wb') as file:
    pickle.dump(iter_map_dict, file)

# Building the second model: LogisticRegression with the weight of evidence features
logger.info('Building the second model')

pipe_model = Pipeline([('preprocess', WoeEncoder()), 
                       ('model', LogisticRegression(C=0.05))])

skf = StratifiedKFold(n_splits=5, random_state=13, shuffle=True)

cv_woe = cross_validate(estimator=pipe_model,
                        X=X_train[cat_col_list],
                        y=y_train['credit_risk'],
                        cv=skf,
                        scoring='roc_auc',
                        return_estimator=True)

logger.info(f"Average ROC-AUC for validation set is {np.round(cv_woe['test_score'].mean(), 4)}.")

# Computing ROC-AUC for the test set
test_lr_woe_ppred_list = []
for i_model in cv_woe['estimator']:
    ppred_y_test = i_model.predict_proba(X_test[cat_col_list])[:, 1]
    test_lr_woe_ppred_list.append(ppred_y_test.reshape(-1, 1))

test_ppred_woe = np.hstack(test_lr_woe_ppred_list).mean(axis=1)
roc_test_woe = roc_auc_score(y_test['credit_risk'], test_ppred_woe)
roc_test_woe = np.round(roc_test_woe, 4)

logger.info(f"ROC-AUC of test set is {roc_test_woe}.")

# Computing recall for the test set
fpr, tpr, thresholds = roc_curve(y_test, test_ppred_woe)
opt_threshold = thresholds[np.argmax(tpr - fpr)]

recall_test_woe = np.round(tpr[np.argmax(tpr-fpr)], 4)

logger.info(f"Recall of test set is {recall_test_woe}.")

# Saving the model for deployment
logger.info('Saving the model for deployment')

with open('woe_cv.pkl', 'wb') as file:
    pickle.dump(cv_woe, file)

# Building the third model: DecisionTree with the all features
logger.info('Building the third model')

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)

skf = StratifiedKFold(n_splits=5, random_state=13, shuffle=True)

cv_dt = cross_validate(estimator=dt_model,
                       X=X_train[all_feat_list],
                       y=y_train['credit_risk'],
                       cv=skf,
                       scoring='roc_auc',
                       return_estimator=True)

logger.info(f"Average ROC-AUC for validation set is {np.round(cv_dt['test_score'].mean(), 4)}.")

# Computing ROC-AUC for the test set
test_dt_ppred_list = []
for i_model in cv_dt['estimator']:
    ppred_y_test = i_model.predict_proba(X_test[all_feat_list])[:, 1]
    test_dt_ppred_list.append(ppred_y_test.reshape(-1, 1))

test_ppred_dt = np.hstack(test_dt_ppred_list).mean(axis=1)
roc_test_dt = roc_auc_score(y_test['credit_risk'], test_ppred_dt)
roc_test_dt = np.round(roc_test_dt, 4)

logger.info(f"ROC-AUC of test set is {roc_test_dt}.")

# Computing recall for the test set
fpr, tpr, thresholds = roc_curve(y_test, test_ppred_dt)
opt_threshold = thresholds[np.argmax(tpr - fpr)]

recall_test_dt = np.round(tpr[np.argmax(tpr-fpr)], 4)

logger.info(f"Recall of test set is {recall_test_dt}.")

# Saving the model for deployment
logger.info('Saving the model for deployment')

with open('dt_cv.pkl', 'wb') as file:
    pickle.dump(cv_dt, file)

# Computing ensemble prediction probabilities
logger.info('Creating an ensemble of all models')

test_ppred_all = (test_ppred_woe + test_ppred_ohe + test_ppred_dt) / 3

# Computing ROC-AUC for the test set using the ensemble approach
roc_test_all = roc_auc_score(y_test['credit_risk'], test_ppred_all)
roc_test_all = np.round(roc_test_all, 4)

logger.info(f"ROC-AUC of test set is {roc_test_all}.")

# Computing recall for the test set using the ensemble approach
fpr, tpr, thresholds = roc_curve(y_test, test_ppred_all)
opt_threshold = thresholds[np.argmax(tpr - fpr)]

recall_test_all = np.round(tpr[np.argmax(tpr-fpr)], 4)

logger.info(f"Recall of test set is {recall_test_all}.")