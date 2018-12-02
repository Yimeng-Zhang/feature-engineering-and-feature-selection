# Feature Engineering & Feature Selection

## About

A comprehensive [guide]() for **Feature Engineering** and **Feature Selection**, with implementations and examples in Python.



## What You'll Learn

Not only a collection of hands-on functions, but also explanation on  **Why**, **How** and **When** to adopt **Which** techniques of feature engineering in data mining. 

- the nature and risk of data problem we often encounter
- explanation of the various feature engineering & selection techniques
- rationale to use it
- pros & cons of each method 
- code & example



## Getting Started

This repo is mainly used as a reference for anyone who are doing feature engineering, and most of the modules are implemented through scikit-learn or its communities.

To run the demos or use the customized function,  please download the ZIP file from the repo or just copy-paste any part of the code you find helpful. They should all be very easy to understand.

**Required Dependencies**:

- Python 3.5, 3.6 or 3.7
- numpy>=1.15
- pandas>=0.23
- scipy>=1.1.0
- scikit_learn>=0.20.1
- seaborn>=0.9.0



## Table of Contents and Code Examples

Below is a list of methods currently implemented in the repo. The complete guide can be found [here]().

**1. Data Exploration**

   1.1 Variables 
   1.2 Variable Identification   
   ​          Check Data Types
   1.3 Univariate Analysis
   ​          Descriptive Analysis
   ​          Discrete Variable Barplot
   ​          Discrete Variable Countplot
   ​          Discrete Variable Boxplot
   ​          Continuous Variable Distplot
   1.4 Bi-variate Analysis
   ​          Scatter Plot
   ​          Correlation Plot
   ​          Heat Map

**2. Feature Cleaning**

   2.1 Missing Values
   ​          Missing Value Check
   ​          Listwise Deletion
   ​          Mean/Median/Mode Imputation
   ​          End of distribution Imputation
   ​          Random Imputation
   ​          Arbitrary Value Imputation
   ​          Add a variable to denote NA
   2.2 Outliers
   ​          Detect by Arbitrary Boundary
   ​          Detect by Mean & Standard Deviation
   ​          Detect by IQR 
   ​          Detect by MAD   
   ​          Mean/Median/Mode Imputation
   ​          Discretization
   ​          Imputation with Arbitrary Value
   ​          Windsorization
   ​          Discard Outliers
   2.3 Rare Values
   ​          Mode Imputation  
   ​          Grouping into One New Category
   2.4 High Cardinality
   ​          Grouping Labels with Business Understanding 
   ​          Grouping Labels with Rare Occurrence into One Category
   ​          Grouping Labels with Decision Tree

**3. Feature Engineering**

   3.1 Feature Scaling  
   ​          Normalization - Standardization 
   ​          Min-Max Scaling
   ​          Robust Scaling
   3.2 Discretize   
   ​          Equal Width Binning
   ​          Equal Frequency Binning
   ​          K-means Binning   
   ​          Discretization by Decision Trees
   ​          ChiMerge
   3.3 Feature Encoding
   ​          One-hot Encoding
   ​          Ordinal-Encoding
   ​          Count/frequency Encoding 
   ​          Mean Encoding
   ​          WOE Encoding
   ​          Target Encoding
   3.4 Feature Transformation
   ​          Logarithmic Transformation
   ​          Reciprocal Transformation
   ​          Square Root Transformation
   ​          Exponential Transformation
   ​          Box-cox Transformation
   ​          Quantile Transformation
   3.5 Feature Generation
   ​          Missing Data Derived
   ​          Simple Stats
   ​          Crossing
   ​          Ratio & Proportion
   ​          Cross Product
   ​          Polynomial
   ​          Feature Leanring by Tree
   ​          Feature Leanring by Deep Network

**4. Feature Selection**

   4.1 Filter Method
   ​          Variance
   ​          Correlation
   ​          Chi-Square
   ​          Mutual Information Filter
   ​          Univariate ROC-AUC or MSE
   ​          Information Value (IV)
   4.2 Wrapper Method
   ​          Forward Selection
   ​          Backward Elimination
   ​          Exhaustive Feature Selection
   ​          Genetic Algorithm
   4.3 Embedded Method
   ​          Lasso (L1)
   ​          Random Forest Importance
   ​          Gradient Boosted Trees Importance
   4.4 Feature Shuffling
   ​          Random Shuffling
   4.5 Hybrid Method
   ​          Recursive Feature Selection 
   ​          Recursive Feature Addition




## Motivation

Feature Engineering & Selection is the most essential part of building a useable machine learning project, even though hundreds of cutting-edge machine learning algorithms coming in these days like deep learning and transfer learning. Indeed, like what Prof Domingos, the author of  *'The Master Algorithm'* says:

> “At the end of the day, some machine learning projects succeed and some fail. What makes the difference? Easily the most important factor is the features used.”
>
> — Prof. Pedro Domingos

![001](./images/001.png)
Data and feature determine the upper limit of a ML project, while models and algorithms are just approaching that limit. However, few materials could be found that systematically introduce the art of feature engineering, and even fewer could explain the rationale behind. This repo aims at teaching you a good guide for Feature Engineering & Selection.



## Key Links and Resources

- Udemy's Feature Engineering online course

https://www.udemy.com/feature-engineering-for-machine-learning/

- Udemy's Feature Selection online course

https://www.udemy.com/feature-selection-for-machine-learning

- JMLR Special Issue on Variable and Feature Selection

http://jmlr.org/papers/special/feature03.html

- Data Analysis Using Regression and Multilevel/Hierarchical Models, Chapter 25: Missing data

http://www.stat.columbia.edu/~gelman/arm/missing.pdf

- Data mining and the impact of missing data

http://core.ecu.edu/omgt/krosj/IMDSDataMining2003.pdf

- PyOD: A Python Toolkit for Scalable Outlier Detection

https://github.com/yzhao062/pyod

- Weight of Evidence (WoE) Introductory Overview

http://documentation.statsoft.com/StatisticaHelp.aspx?path=WeightofEvidence/WeightofEvidenceWoEIntroductoryOverview

- About Feature Scaling and Normalization

http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

- Feature Generation with RF, GBDT and Xgboost

https://blog.csdn.net/anshuai_aw1/article/details/82983997

- A review of feature selection methods with applications

https://ieeexplore.ieee.org/iel7/7153596/7160221/07160458.pdf


