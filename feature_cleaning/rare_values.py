import pandas as pd
# import numpy as np
# from warnings import warn

# 2018.11.07 Created by Eamon.Zhang
# 2018.11.12 change into fit() transform() format

class GroupingRareValues():
    """
    Grouping the observations that show rare labels into a unique category ('rare')
    
    Parameters
    ----------
   
    """

    def __init__(self, mapping=None, cols=None, threshold=0.01):
        self.cols = cols
        self.mapping = mapping
        self._dim = None
        self.threshold = threshold


    def fit(self, X, y=None, **kwargs):
        """Fit encoder according to X and y.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : encoder
            Returns self.
        """

        self._dim = X.shape[1]

        _, categories = self.grouping(
            X,
            mapping=self.mapping,
            cols=self.cols,
            threshold=self.threshold
        )
        self.mapping = categories
        return self


    def transform(self, X):
        """Perform the transformation to new categorical data.
        Will use the mapping (if available) and the column list to encode the
        data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        X : Transformed values with encoding applied.
        """

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        #  make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        X, _ = self.grouping(
            X,
            mapping=self.mapping,
            cols=self.cols,
            threshold=self.threshold
        )

        return X 


    def grouping(self, X_in, threshold, mapping=None, cols=None):
        """
        Grouping the observations that show rare labels into a unique category ('rare')

        """

        X = X_in.copy(deep=True)

#        if cols is None:
#            cols = X.columns.values

        if mapping is not None:  # transform
            mapping_out = mapping
            for i in mapping:
                column = i.get('col') # get the column name
                X[column] = X[column].map(i['mapping'])

#                try:
#                    X[column] = X[column].astype(int)
#                except ValueError as e:
#                    X[column] = X[column].astype(float)
        else: # fit
            mapping_out = []
            for col in cols:
#                if util.is_category(X[col].dtype):
#                    categories = X[col].cat.categories
#                else:
                temp_df = pd.Series(X[col].value_counts()/len(X))
                mapping = { k: ('rare' if k not in temp_df[temp_df >= threshold].index else k)
                          for k in temp_df.index}

                mapping = pd.Series(mapping)
                mapping_out.append({'col': col, 'mapping': mapping, 'data_type': X[col].dtype}, )

        return X, mapping_out



#==============================================================================
# def rare_imputation(X_train, X_test, variable):
#     
#     # find the most frequent category
#     frequent_cat = X_train.groupby(variable)[variable].count().sort_values().tail(1).index.values[0]
#     
#     # find rare labels
#     temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))
#     rare_cat = [x for x in temp.loc[temp<0.05].index.values]
#     
#     # create new variables, with Rare labels imputed
#     
#     # by the most frequent category
#     X_train[variable+'_freq_imp'] = np.where(X_train[variable].isin(rare_cat), frequent_cat, X_train[variable])
#     X_test[variable+'_freq_imp'] = np.where(X_test[variable].isin(rare_cat), frequent_cat, X_test[variable])
#     
#     # by adding a new label 'Rare'
#     X_train[variable+'_rare_imp'] = np.where(X_train[variable].isin(rare_cat), 'Rare', X_train[variable])
#     X_test[variable+'_rare_imp'] = np.where(X_test[variable].isin(rare_cat), 'Rare', X_test[variable])
#==============================================================================

# 2018.11.26 created by Eamon.Zhang
class ModeImputation():
    """
    Replacing the rare label by most frequent label
    
    Parameters
    ----------
   
    """

    def __init__(self, mapping=None, cols=None, threshold=0.01):
        self.cols = cols
        self.mapping = mapping
        self._dim = None
        self.threshold = threshold


    def fit(self, X, y=None, **kwargs):
        """Fit encoder according to X and y.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : encoder
            Returns self.
        """

        self._dim = X.shape[1]

        _, categories = self.impute_with_mode(
            X,
            mapping=self.mapping,
            cols=self.cols,
            threshold=self.threshold
        )
        self.mapping = categories
        return self


    def transform(self, X):
        """Perform the transformation to new categorical data.
        Will use the mapping (if available) and the column list to encode the
        data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        X : Transformed values with encoding applied.
        """

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        #  make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        X, _ = self.impute_with_mode(
            X,
            mapping=self.mapping,
            cols=self.cols,
            threshold=self.threshold
        )

        return X 


    def impute_with_mode(self, X_in, threshold, mapping=None, cols=None):
        """
        Grouping the observations that show rare labels into a unique category ('rare')

        """

        X = X_in.copy(deep=True)

#        if cols is None:
#            cols = X.columns.values

        if mapping is not None:  # transform
            mapping_out = mapping
            for i in mapping:
                column = i.get('col') # get the column name
                X[column] = X[column].map(i['mapping'])

#                try:
#                    X[column] = X[column].astype(int)
#                except ValueError as e:
#                    X[column] = X[column].astype(float)
        else: # fit
            mapping_out = []
            for col in cols:
#                if util.is_category(X[col].dtype):
#                    categories = X[col].cat.categories
#                else:
                temp_df = pd.Series(X[col].value_counts()/len(X))
                median = X[col].mode()[0]
                mapping = { k: (median if k not in temp_df[temp_df >= threshold].index else k)
                          for k in temp_df.index}

                mapping = pd.Series(mapping)
                mapping_out.append({'col': col, 'mapping': mapping, 'data_type': X[col].dtype}, )

        return X, mapping_out
