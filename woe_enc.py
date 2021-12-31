import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def get_feat_power(x):
    if x < 0.02:
        return 'Useless'
    elif x >= 0.02 and x < 0.1:
        return 'Weak predictors'
    elif x >= 0.1 and x < 0.3:
        return 'Medium predictors'
    elif x >= 0.3 and x <= 0.5:
        return 'Strong predictors'
    else:
        return 'Suspicious'

class WoeEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y):
        temp_dev_df = pd.concat([X, y], axis=1)
        target_col = y.name
        temp_feat = []
        temp_map_dict = {}
        for col in X.columns:
            # if col == 'amount_dage_iter':
            #     continue
            temp_gdev_df = temp_dev_df.groupby(col)[target_col].agg(['count', 'sum'])
            temp_gdev_df.rename(columns={'sum': 'event'}, inplace=True)
            temp_gdev_df['non_event'] = temp_gdev_df['count'] - temp_gdev_df['event']
            temp_gdev_df['p_event'] = temp_gdev_df['event'] / temp_gdev_df['event'].sum()
            temp_gdev_df['p_non_event'] = temp_gdev_df['non_event'] / temp_gdev_df['non_event'].sum()
            temp_gdev_df['woe'] = np.log(temp_gdev_df['p_event'] / temp_gdev_df['p_non_event'])
            temp_gdev_df['iv'] = temp_gdev_df['woe'] * (temp_gdev_df['p_event'] - temp_gdev_df['p_non_event'])
            if get_feat_power(x=temp_gdev_df['iv'].sum()) in ['Medium predictors', 'Strong predictors']:
                temp_feat.append(col)
                temp_map_dict[col] = temp_gdev_df['woe'].to_dict()
        self.temp_feat = temp_feat
        self.temp_map_dict = temp_map_dict
        return self
    def transform(self, X):
        for col in self.temp_feat:
            X[col] = X[col].map(self.temp_map_dict[col])
        return X[self.temp_feat]