import pandas as pd

# 2018.11.28 Created by Eamon.Zhang

class MeanEncoding():
    """
    replacing the label by the mean of the target for that label. 
    
    Parameters
    ----------
   
    """

    def __init__(self, mapping=None, cols=None):
        self.cols = cols
        self.mapping = mapping
        self._dim = None
        # self.threshold = threshold


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

        _, categories = self.mean_encoding(
            X,
            y,
            mapping=self.mapping,
            cols=self.cols
            # threshold=self.threshold
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

        X, _ = self.mean_encoding(
            X,
            mapping=self.mapping,
            cols=self.cols
            # threshold=self.threshold
        )

        return X 


    def mean_encoding(self, X_in, y=None, mapping=None, cols=None):
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
                mapping = X[y.name].groupby(X[col]).mean().to_dict()
                mapping = pd.Series(mapping)
                mapping_out.append({'col': col, 'mapping': mapping, 'data_type': X[col].dtype}, )

        return X, mapping_out