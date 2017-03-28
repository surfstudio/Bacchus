from VerboseConfigurer import VerboseConfigurer
from sklearn.base import BaseEstimator
from sklearn.pipeline import TransformerMixin, Pipeline, FeatureUnion, Pipeline, \
                                _transform_one, _fit_transform_one
from sklearn.externals.joblib import Parallel, delayed


class DFPipeline(Pipeline, VerboseConfigurer):
    def __init__(self, steps, verbose=False):
        super().__init__(steps)
        self.verbose = verbose
        self._apply_verbose(self.verbose)

    def fit_transform(self, X, y=None, **fit_params):
        if self.verbose:
            print('%s%s\t%s' % (self.__class__.__name__.ljust(25), now(), X.shape))
        return super().fit_transform(X, y, **fit_params)


class DFFeatureUnion(FeatureUnion, VerboseConfigurer):
    def __init__(self, transformer_list, prefix=None, drop=False, n_jobs=1, transformer_weights=None):
        super().__init__(transformer_list, n_jobs, transformer_weights)
        self.drop = drop
        self.prefix = prefix

    def _rename_columns_if_needed(self, df, excepting=None):
        if self.prefix is not None:
            r = {}; [r.update({x: self.prefix + '_' + x}) for x in df.columns.values
                     if excepting is None or x not in excepting]
            df.rename(columns=r, inplace=True)

    def _concat_just_right(self, X, to_concat):
        X_cols = X.columns.values
        result = X
        for another in to_concat:
            if another is None or another.shape == (0,0):
                continue
            cols = another.columns.values
            if len(set(cols).intersection(X_cols)) == 0:
                assert(result.shape[0] == another.shape[0]), 'Number of rows must be the same'
                self._rename_columns_if_needed(another)
                result = pd.concat((result, another), axis=1)
            else:
                self._rename_columns_if_needed(another, excepting=set(cols).intersection(X_cols))
                result = pd.merge(result, another, how='left')
        return result

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, name, weight, X, y, **fit_params)
            for name, trans, weight in self._iter())
        to_concats = [r[0] for r in result]
        if self.drop:
            return self._concat_just_right(to_concats[0], to_concats[1:])
        return self._concat_just_right(X, to_concats)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, name, weight, X)
            for name, trans, weight in self._iter())
        if self.drop:
            return self._concat_just_right(Xs[0], Xs[1:])
        return self._concat_just_right(X, Xs)


class DFConcat(FeatureUnion, VerboseConfigurer):
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, name, weight, X, y, **fit_params)
            for name, trans, weight in self._iter())
        Xs = [r[0] for r in result]
        Xs.insert(0, X)
        return pd.concat(Xs, axis=0)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, name, weight, X)
            for name, trans, weight in self._iter())
        Xs.insert(0, X)
        return pd.concat(Xs, axis=0)


class AbstractTransformer(BaseEstimator):
    def __init__(self, columns_include=[], columns_exclude=[], \
                       save_to_cache=False, load_cache=False):
        assert(not(len(columns_include) > 0 and len(columns_exclude) > 0)), \
            "Can't specify both include and exclude"
        self._columns_include = columns_include
        self._columns_exclude = columns_exclude
        self._save = save_to_cache
        self._load = load_cache
        self.verbose = False

    @property
    def class_name(self):
        return self.__class__.__name__

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if self._load:
            if self.verbose:
                print('%s%s\t%s' % (self.class_name.ljust(25), now(), X.shape))
            return self._load_pickle()
        data = None
        if y is None:
            data = self.fit(X, **fit_params).transform(X)
        else:
            data = self.fit(X, y, **fit_params).transform(X)
        if self._save:
            self._save_pickle(data)
        if self.verbose:
            print('%s%s\t%s' % (self.class_name.ljust(25), now(), data.shape))
        return data

    def transform(self, X, **transfrom_params):
        raise Exception('Implement this method!')

    # --- Better columns accessors ---

    def _columns_to_apply(self, X):
        if len(self._columns_include) == 0 and len(self._columns_exclude) == 0:
            return None
        x_columns = X.columns.values
        if len(self._columns_include) > 0:
            result = [x for x in self._columns_include if x in x_columns]
        else:
            result = [x for x in x_columns if x not in self._columns_exclude]
        return X[result]

    def _numeric_columns(self, X):
        return X.select_dtypes(include=['number'])

    def _nonnumeric_columns(self, X):
        return X.select_dtypes(exclude=['number'])

    def _rest_columns(self, X, df):
        return X[[c for c in X.columns.values if c not in df.columns.values]]

    # --- Pickling ---
    def _save_pickle(self, X, name=None):
        import pickle
        if name is None:
            name = "%s.pkl" % self.class_name
        if not name.endswith('.pkl'):
            name += '.pkl'
        with open(name, 'wb') as handle:
            pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_pickle(self, filename=None):
        import pickle
        if filename is None:
            filename = '%s.pkl' % self.class_name
        with open(filename, 'rb') as handle:
            return pickle.load(handle)
