import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
import cPickle as pickle


def label_creation(data):
    '''
    INPUT: dataframe
    OUTPUT: dataframe

    Determines which suppliers have been offerd early payment, which have accepted,
    and the percentage of EP spend dollars are taken.
    '''

    data['EP_Offered'] = data['Early Payment Spend Offered One Year'].apply(lambda x: 1 if x>0 else 0)
    data['EP_Accepted'] = data['Early Payment Spend Taken One Year'].apply(lambda x: 1 if x>0 else 0)
    data['Participation_Rate'] = 1.*(data['Early Payment Spend Taken One Year']/
                                data['Early Payment Spend Offered One Year'])
    return data


def data_imputation(data):
    '''
    INPUT: dataframe
    OUTPUT: dataframe

    Imputes -1B for missing revenue values; may want to impute others later.
    '''

    data['Annual Revenue'].fillna(-1000000000, inplace=True)
    return data


def billing_countries(data):
    '''
    INPUT: dataframe
    OUTPUT: dataframe

    Breaks out dummy variables for the billing country if the country is one of
    US, CA, GB, or DE
    '''

    data['US'] = data['Billing Country'].apply(lambda x: 1 if x=='US' else 0)
    data['CA'] = data['Billing Country'].apply(lambda x: 1 if x=='CA' else 0)
    data['GB'] = data['Billing Country'].apply(lambda x: 1 if x=='GB' else 0)
    data['DE'] = data['Billing Country'].apply(lambda x: 1 if x=='DE' else 0)
    data.drop(['Billing Country'], axis=1, inplace=True)
    return data


def data_filtering(data):
    '''
    INPUT: dataframe
    OUTPUT: dataframe

    Filtering out relations where EPs have not been offered and account type is DPT.
    '''

    data = data[(data['EP_Offered']==1)&(data['DPT EP Spend Taken One Year']==0)]
    data = data[(pd.notnull(data['Annual Revenue'])) &
            (pd.notnull(data['Early Payment Rate of Last EP Offered']))][[
           u'Enrolled', u'Invoice Spend Last One Year',
           u'Invoice Count Last One Year',
        u'Annual Revenue',
       u'Inv Average Term Days Last 30 Days',
       'Inv Average Days to Approval 30 Days',
         u'Inv Average Days To Pay 30 Days',
        u'US', u'CA', u'GB', u'DE',
        u'Early Payment Rate of Last EP Offered',
         'Participation_Rate']]
    return data


def grid_search():
    '''
    INPUT:
    OUTPUT: estimator

    Runs a grid search for a GBM and returns the best one.
    '''

    gd_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
           'max_depth': [4, 8, 12],
           'min_samples_leaf': [5,10,15],
           'max_features': [1.0, 0.3, 0.1],
           'n_estimators': [100, 500, 1000]}

    grid_cv = GridSearchCV(GradientBoostingRegressor(), gd_grid, n_jobs=-1, verbose=True,
                               scoring='mean_squared_error').fit(X_train, y_train)
    return grid_cv.best_estimator_


if __name__ == '__main__':

    data = pd.read_csv('/Users/tracy.lee/Downloads/Supplier Details Platform 160929.csv')
    label_creation(data)
    data_imputation(data)
    billing_countries(data)
    data = data_filtering(data)

    y = data.pop('Participation_Rate').values
    X = data.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # params = grid_search()
    # print params


    gbr = GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.01, loss='ls',
             max_depth=10, max_features=0.3, max_leaf_nodes=None,
             min_samples_leaf=10, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=1000,
             presort='auto', random_state=None, subsample=1.0, verbose=0,
             warm_start=False)
    gbr.fit(X_train,y_train)

    # gbr_scores = cross_val_score(gbr, X_train, y_train,scoring ='mean_squared_error', cv=5)
    # print gbr_scores

    pickle.dump(gbr, open('data/participation_model.pkl', 'wb') )
