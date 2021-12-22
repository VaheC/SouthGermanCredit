from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import pandas as pd
import numpy as np
from woe_enc import WoeEncoder
import sklearn

app = Flask(__name__)

with open('ohe_transformer.pkl', 'rb') as file:
    ohe_transformer = pickle.load(file)

with open('ohe_cv.pkl', 'rb') as file:
    cv_ohe = pickle.load(file)

with open('kbd_dict.pkl', 'rb') as file:
    kbd_dict = pickle.load(file)

with open('iter_map_dict.pkl', 'rb') as file:
    iter_map_dict = pickle.load(file)

with open('woe_cv.pkl', 'rb') as file:
    cv_woe = pickle.load(file)

with open('dt_cv.pkl', 'rb') as file:
    cv_dt = pickle.load(file)

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        status = int(str(request.form['status']).split(':')[0].strip())
        duration = int(request.form['duration'])
        credit_history = int(str(request.form['credit_history']).split(':')[0].strip()) - 1
        purpose = int(str(request.form['purpose']).split(':')[0].strip()) - 1
        amount = float(request.form['amount'])
        savings = int(str(request.form['savings']).split(':')[0].strip())
        employment_duration = int(str(request.form['employment_duration']).split(':')[0].strip())
        dti = int(str(request.form['dti']).split(':')[0].strip())
        status_sex = int(str(request.form['status_sex']).split(':')[0].strip())
        other_debtors = int(str(request.form['other_debtors']).split(':')[0].strip())
        present_residence = int(str(request.form['present_residence']).split(':')[0].strip())
        mv_property = int(str(request.form['mv_property']).split(':')[0].strip())
        age = int(request.form['age'])
        other_installment_plans = int(str(request.form['other_installment_plans']).split(':')[0].strip())
        housing = int(str(request.form['housing']).split(':')[0].strip())
        number_credits = int(str(request.form['number_credits']).split(':')[0].strip())
        job = int(str(request.form['job']).split(':')[0].strip())
        people_liable = int(str(request.form['people_liable']).split(':')[0].strip())
        telephone = int(str(request.form['telephone']).split(':')[0].strip())
        foreign_worker = int(str(request.form['foreign_worker']).split(':')[0].strip())

        # Let's create a function which will allow to assign a datapoint 
        # to one the clusters derived during the analysis.
        def get_hcluster(x):
            # The hard coded values below come from the model_estimation.ipynb.
            # The values represent medians for age, amount, and duration 
            # respectively in each cluster.
            h1 = np.array([36, 4736, 30])
            h2 = np.array([49, 1264, 12])
            h3 = np.array([29, 1599.5, 12])
            dh1 = np.linalg.norm(x-h1)
            dh2 = np.linalg.norm(x-h2)
            dh3 = np.linalg.norm(x-h3)
            dh_list = [dh1, dh2, dh3]
            min_index = dh_list.index(np.min(dh_list))
            cluster_index = min_index + 1
            return cluster_index

        x = np.array([age, amount, duration])
        hclusters = get_hcluster(x)

        dage = (100 - age) / (12 * duration)

        # Let's create a dictionary of lmbda values (Box-Cox transformation lmbda in scipy)
        # for each numeric feature. The values come from the model_estimation.ipynb.
        lmbda_dict = {'age': -0.6524316739182968,
                      'amount': -0.0639326907261038,
                      'duration': 0.09297575561665981,
                      'dage': -0.025199226092440907}

        # Now let's apply Box-Cox transformation on numeric features.
        age = (age**lmbda_dict['age'] - 1) / lmbda_dict['age']
        dage = (dage**lmbda_dict['dage'] - 1) / lmbda_dict['dage']
        amount = (amount**lmbda_dict['amount'] - 1) / lmbda_dict['amount']
        duration = (duration**lmbda_dict['duration'] - 1) / lmbda_dict['duration']


        data = pd.DataFrame([job, employment_duration, number_credits, 
                             other_debtors, status_sex, foreign_worker, 
                             status, credit_history, people_liable, dti, 
                             savings, telephone, mv_property, purpose, 
                             other_installment_plans, housing, 
                             present_residence, hclusters]).T
        cat_col_list = ['job', 'employment_duration', 'number_credits', 
                        'other_debtors', 'status_sex', 'foreign_worker', 
                        'status', 'credit_history', 'people_liable', 'dti', 
                        'savings', 'telephone', 'property', 'purpose', 
                        'other_installment_plans', 'housing', 
                        'present_residence', 'hclusters']
        data.columns = cat_col_list

        num_data = pd.DataFrame([age, amount, duration, dage]).T
        num_col_list = ['age', 'amount', 'duration', 'dage']
        num_data.columns = num_col_list

        data_ohe = pd.DataFrame(ohe_transformer.transform(data[cat_col_list]))
        data_ohe.columns = list(ohe_transformer.get_feature_names_out(cat_col_list))
        data_ohe = pd.concat([num_data, data_ohe], axis=1)

        lr_ohe_ppred_list = []
        for i_model in cv_ohe['estimator']:
            lr_ohe_ppred = i_model.predict_proba(data_ohe)[:, 1]
            lr_ohe_ppred_list.append(lr_ohe_ppred.reshape(-1, 1))

        ppred_ohe = np.hstack(lr_ohe_ppred_list).mean(axis=1)

        data = pd.concat([num_data, data], axis=1)
        for col in num_col_list:
            data[f'{col}_bin'] = kbd_dict[col].transform(data[col].to_frame())
            cat_col_list.append(f'{col}_bin')

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

        for i_list in iter_list:
            temp_iter_feat_name = '_'.join(i_list) + '_iter'
            data[temp_iter_feat_name] = data[i_list[0]].astype(str) + \
                                        '_' + data[i_list[1]].astype(str)
            data[temp_iter_feat_name] = data[temp_iter_feat_name].\
                                        map(iter_map_dict[iter_list.index(i_list)])

            cat_col_list.append(temp_iter_feat_name)

        lr_woe_ppred_list = []
        for i_model in cv_woe['estimator']:
            lr_woe_ppred = i_model.predict_proba(data[cat_col_list])[:, 1]
            lr_woe_ppred_list.append(lr_woe_ppred.reshape(-1, 1))

        ppred_woe = np.hstack(lr_woe_ppred_list).mean(axis=1)

        dt_feat_list = num_col_list.copy()
        dt_feat_list.extend(cat_col_list)

        dt_ppred_list = []
        for i_model in cv_dt['estimator']:
            dt_ppred = i_model.predict_proba(data[dt_feat_list])[:, 1]
            dt_ppred_list.append(dt_ppred.reshape(-1, 1))

        ppred_dt = np.hstack(dt_ppred_list).mean(axis=1)

        ppred = (ppred_woe + ppred_ohe + ppred_dt) / 3
        ppred = round(ppred[0], 2)

        if ppred > 0.27562525642456126:
            decision_text = f'Refuse: probability of default is {100*ppred} %.'
        else:
            decision_text = f'Proceed: probability of default is {100*ppred} %.'
        
        return render_template('index.html', prediction_text=decision_text)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)