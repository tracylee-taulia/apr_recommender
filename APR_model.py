import cPickle as pickle
import pandas as pd
import numpy as np
from participation_rate_model import billing_countries
import re
from scipy.optimize import curve_fit

# names of all the fields in the supplier data template comprehensive version
template_fields = ['Vendor ID*', 'Vendor Name*',  'Annual PO Count',
'Annual Payment Count', 'Payment Term*', 'Payment Term Description*',
'Annual Discount Capture*', 'Currency*', 'Actual Payment Day Avg',
'Avg Invoice Approval Days*', 'Method of Payment', 'Email Address*',
'Physical Address', 'City', 'State', 'Zip/Postal', 'Country*', 'Phone #',
'Fax #', 'Industry*', 'Company Code']

#range of APRs that the model will test; currently 10-27
aprs = 1.*np.arange(40,109)/4

def temporary_enrolled(data):
    '''
    INPUT: dataframe
    OUTPUT: dataframe

    Until I figure out how to actually figure out if the suppliers are already enrolled,
    this is a stopgap measure for randomly selecting the data to be 0 or 1.
    '''
    data['Enrolled'] = np.random.choice(range(0, 2), data.shape[0])
    return data


def net_term_regex(string):
    '''
    INPUT: string
    OUTPUT: int

    Reads in the net terms of the supplier and parses for the number of days.
    '''

    return int(re.findall(r'\d+', string)[0])


def data_transformation(data):
    '''
    INPUT: dataframe
    OUTPUT: dataframe

    Takes the raw upload dataframe and fills missing values with -1B and formats
    the dataframe to the column ordering and format that the trained model is
    expecting.
    '''

    data.fillna(-10000000, inplace=True)
    data.rename(columns={'Country*':'Billing Country'}, inplace=True)
    billing_countries(data)
    data = data[[
           u'Enrolled', 'Annual Spend*', 'Annual Invoice Count*',
        u'Annual Revenue',
       'Payment Term*',
       'Avg Invoice Approval Days*',
         'Actual Payment Day Avg',
        u'US', u'CA', u'GB', u'DE']]

    data['Payment Term*'] = data['Payment Term*'].apply(lambda x: net_term_regex(x))
    return data


def APR_trial(model, aprs, arr):
    '''
    INPUT: model, numpy array, numpy array
    OUTPUT: list

    Iterates through each possible APR and runs the usage rate/participation prediction
    for that APR. Returns the results of each trial, i.e. what the participation rate is
    at 10%, 11%, etc.
    '''

    results = []
    for i, apr in enumerate(aprs):
        results.append(model.predict(np.append(arr, apr))[0])
    return results

def fit_curve(aprs, results):
    '''
    INPUT: numpy array, list
    OUTPUT: numpy array, numpy array

    Fits a 5th degree polynomial to the results of the APR trial.
    5 degrees was not rigorously selected for any specific reason.
    '''

    z = np.polyfit(aprs, results, 5)
    f = np.poly1d(z)
    return z, f


def calculate_pr_from_curve(z,x):
    '''
    INPUT: numpy array, float
    OUTPUT: float

    Takes coefficients from the polynomial curve and calculates f(x) for the stated x.
    '''

    return z[0]*x**5 + z[1]*x**4 + z[2]*x**3 + z[3]*x**2 + z[4]*x + z[5]


def plot_curve(aprs, results, f):
    '''
    INPUT: numpy array, list, numpy array
    OUTPUT: graph

    Plots the points used to fit the curve and the fitted curve.
    '''

    x_new = np.linspace(aprs[0], aprs[-1], 21)
    y_new = f(x_new)
    plt.plot(aprs,results,'o',x_new,y_new)
    plt.show()


def optimize_apr_pr(data, model):
    '''
    INPUT: dataframe, model
    OUTPUT: nx2 numpy array

    Runs the APR trial for each row of the data and outputs the highest combination
    of the APR and participation
    '''
    num_records = data.shape[0]
    output = np.empty((num_records, 2))
    for i in xrange(num_records):
        results = APR_trial(model, aprs, data.iloc[i].values)
        z, f = fit_curve(aprs, results)
        opti_func = [x*calculate_pr_from_curve(z, x) for x in aprs]
        idx_highest_combo = np.argsort(opti_func)[-1]
        best_apr = aprs[idx_highest_combo]
        best_pr = calculate_pr_from_curve(z, best_apr)
        output[i] = [best_apr, best_pr]
    return output



if __name__ == '__main__':
    model = pickle.load(open('data/participation_model.pkl', 'r'))
    df = pd.read_csv('test_data.csv')
    # data = temporary_enrolled(df)
    data = data_transformation(df.copy())
    apr_pr = optimize_apr_pr(data, model)
    df[['Recommended_APR', 'Participation_Rate']] = pd.DataFrame(apr_pr)


'''
the enrolled and annual revenue fields in the toy data are currently from my own population
NOT from a join or cap IQ

'''



'''
things this has to do
- read in the pickled model
- transform the input data to run predictions from the pickled model
- per line in the input data, run predicts from 10-30% APR
- fit a curve for the 21 results: scipy.optimize.curve_fit or numpy.polyfit
- solve optimization problem (analytical solution might suffice)
'''
