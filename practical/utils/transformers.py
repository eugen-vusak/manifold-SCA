from sklearn.base import TransformerMixin, BaseEstimator


class DummyTranformer(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
