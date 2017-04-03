from df_classes import AbstractTransformer

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class ModelTransformer(AbstractTransformer):
    def __init__(self, model, target, grid_search=None, random_search=None, **other):
        assert(not(grid_search is not None and random_search is not None)), \
            "Can't specify both GridSearch and RandomizedSearch"
        super().__init__(**other)
        self.target = target
        self._model = model
        self._grid_search = grid_search
        self._random_search = random_search

    def fit(self, X, *args, **kwargs):
        if self._grid_search:
            model = GridSearchCV(self._model, **self._grid_search)
        elif self._random_search:
            model = RandomizedSearchCV(self._model, **self._random_search)
        else:
            model = self._model

        if self._grid_search is not None:
            self._grid = model
        elif self._random_search is not None:
            self._rnd = model

        assert (self.target in X.columns.values), 'X must contain the target column'
        self._xcols = list(X.columns.values)
        self._xcols.remove(self.target)
        if len(self._columns_exclude) == 0 and len(self._columns_include) > 0:
            self._columns_exclude = list(set(self._xcols) - set(self._columns_include))
        [self._xcols.remove(t) for t in self._columns_exclude]
        x = X[self._xcols]
        y = X[self.target]
        model.fit(x, y, **kwargs)
        return self

    def transform(self, X, **transform_params):
        x = X[self._xcols]
        return pd.DataFrame(self.model.predict(x), columns=[self.estimator_class_name])

    @property
    def model(self):
        if hasattr(self, '_grid'):
            return self._grid
        if hasattr(self, '_rnd'):
            return self._rnd
        return self._model

    @property
    def estimator_class_name(self):
        if hasattr(self.model, 'estimator'):
            return self.model.estimator.__class__.__name__
        return self.model.__class__.__name__


class ModelStacker(AbstractTransformer):
    def __init__(self, weights=None, **other):
        super().__init__(**other)
        self.weights = weights
        if weights is not None:
            self.weights /= np.sum(self.weights)

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.mean(axis=1) if self.weights is None else
                            np.sum([X.ix[:,i] * self.weights[i] \
                             for i in range(X.shape[1])], axis=0))
