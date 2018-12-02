import pandas as pd
import numpy as np
# from warnings import warn

# 2018.11.07 Created by Eamon.Zhang

def outlier_detect_arbitrary(data,col,upper_fence,lower_fence):
    '''
    identify outliers based on arbitrary boundaries passed to the function.
    '''

    para = (upper_fence, lower_fence)
    tmp = pd.concat([data[col]>upper_fence,data[col]<lower_fence],axis=1)
    outlier_index = tmp.any(axis=1)
    print('Num of outlier detected:',outlier_index.value_counts()[1])
    print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))    
    return outlier_index, para



def outlier_detect_IQR(data,col,threshold=3):
    '''
    outlier detection by Interquartile Ranges Rule, also known as Tukey's test. 
    calculate the IQR ( 75th quantile - 25th quantile) 
    and the 25th 75th quantile. 
    Any value beyond:
        upper bound = 75th quantile + （IQR * threshold）
        lower bound = 25th quantile - （IQR * threshold）   
    are regarded as outliers. Default threshold is 3.
    '''
     
    IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
    Lower_fence = data[col].quantile(0.25) - (IQR * threshold)
    Upper_fence = data[col].quantile(0.75) + (IQR * threshold)
    para = (Upper_fence, Lower_fence)
    tmp = pd.concat([data[col]>Upper_fence,data[col]<Lower_fence],axis=1)
    outlier_index = tmp.any(axis=1)
    print('Num of outlier detected:',outlier_index.value_counts()[1])
    print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))
    return outlier_index, para


def outlier_detect_mean_std(data,col,threshold=3):
    '''
    outlier detection by Mean and Standard Deviation Method.
    If a value is a certain number(called threshold) of standard deviations away 
    from the mean, that data point is identified as an outlier. 
    Default threshold is 3.

    This method can fail to detect outliers because the outliers increase the standard deviation. 
    The more extreme the outlier, the more the standard deviation is affected.
    '''
   
    Upper_fence = data[col].mean() + threshold * data[col].std()
    Lower_fence = data[col].mean() - threshold * data[col].std()   
    para = (Upper_fence, Lower_fence)   
    tmp = pd.concat([data[col]>Upper_fence,data[col]<Lower_fence],axis=1)
    outlier_index = tmp.any(axis=1)
    print('Num of outlier detected:',outlier_index.value_counts()[1])
    print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))
    return outlier_index, para


def outlier_detect_MAD(data,col,threshold=3.5):
    """
    outlier detection by Median and Median Absolute Deviation Method (MAD)
    The median of the residuals is calculated. Then, the difference is calculated between each historical value and this median. 
    These differences are expressed as their absolute values, and a new median is calculated and multiplied by 
    an empirically derived constant to yield the median absolute deviation (MAD). 
    If a value is a certain number of MAD away from the median of the residuals, 
    that value is classified as an outlier. The default threshold is 3 MAD.
    
    This method is generally more effective than the mean and standard deviation method for detecting outliers, 
    but it can be too aggressive in classifying values that are not really extremely different. 
    Also, if more than 50% of the data points have the same value, MAD is computed to be 0, 
    so any value different from the residual median is classified as an outlier.
    """
    
    median = data[col].median()
    median_absolute_deviation = np.median([np.abs(y - median) for y in data[col]])
    modified_z_scores = pd.Series([0.6745 * (y - median) / median_absolute_deviation for y in data[col]])
    outlier_index = np.abs(modified_z_scores) > threshold
    print('Num of outlier detected:',outlier_index.value_counts()[1])
    print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))
    return outlier_index


# 2018.11.10 outlier treatment
def impute_outlier_with_arbitrary(data,outlier_index,value,col=[]):
    """
    impute outliers with arbitrary value
    """
    
    data_copy = data.copy(deep=True)
    for i in col:
        data_copy.loc[outlier_index,i] = value
    return data_copy
    
    
def windsorization(data,col,para,strategy='both'):
    """
    top-coding & bottom coding (capping the maximum of a distribution at an arbitrarily set value,vice versa)
    """
    
    data_copy = data.copy(deep=True)  
    if strategy == 'both':
        data_copy.loc[data_copy[col]>para[0],col] = para[0]
        data_copy.loc[data_copy[col]<para[1],col] = para[1]
    elif strategy == 'top':
        data_copy.loc[data_copy[col]>para[0],col] = para[0]
    elif strategy == 'bottom':
        data_copy.loc[data_copy[col]<para[1],col] = para[1]  
    return data_copy


def drop_outlier(data,outlier_index):
    """
    drop the cases that are outliers
    """
    
    data_copy = data[~outlier_index]
    return data_copy


def impute_outlier_with_avg(data,col,outlier_index,strategy='mean'):
    """
    impute outlier with mean/median/most frequent values of that variable.
    """
    
    data_copy = data.copy(deep=True)
    if strategy=='mean':
        data_copy.loc[outlier_index,col] = data_copy[col].mean()
    elif strategy=='median':
        data_copy.loc[outlier_index,col] = data_copy[col].median()
    elif strategy=='mode':
        data_copy.loc[outlier_index,col] = data_copy[col].mode()[0]   
        
    return data_copy
