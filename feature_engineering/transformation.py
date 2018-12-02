import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
# from warnings import warn

# 2018.11.26 Created by Eamon.Zhang
def diagnostic_plots(df, variable):
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist()

    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=pylab)

    plt.show()
    
    
def log_transform(data,cols=[]):
    """
    Logarithmic transformation
    """
    
    data_copy = data.copy(deep=True)
    for i in cols:
        data_copy[i+'_log'] = np.log(data_copy[i]+1)
        print('Variable ' + i +' Q-Q plot')
        diagnostic_plots(data_copy,str(i+'_log'))       
    return data_copy 


def reciprocal_transform(data,cols=[]):
    """
    Reciprocal transformation
    """
    
    data_copy = data.copy(deep=True)
    for i in cols:
        data_copy[i+'_reciprocal'] = 1/(data_copy[i])
        print('Variable ' + i +' Q-Q plot')
        diagnostic_plots(data_copy,str(i+'_reciprocal'))       
    return data_copy 


def square_root_transform(data,cols=[]):
    """
    square root transformation
    """
    
    data_copy = data.copy(deep=True)
    for i in cols:
        data_copy[i+'_square_root'] = (data_copy[i])**(0.5)
        print('Variable ' + i +' Q-Q plot')
        diagnostic_plots(data_copy,str(i+'_square_root'))        
    return data_copy 


def exp_transform(data,coef,cols=[]):
    """
    exp transformation
    """
    
    data_copy = data.copy(deep=True)
    for i in cols:
        data_copy[i+'_exp'] = (data_copy[i])**coef
        print('Variable ' + i +' Q-Q plot')
        diagnostic_plots(data_copy,str(i+'_exp'))         
    return data_copy 

