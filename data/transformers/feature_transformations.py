from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

import category_encoders as ce


class Scaler(AbstractTransformer):
    def __init__(self, feature_range=None, **other):
        super().__init__(**other)
        self._scaler_class = StandardScaler if feature_range is None else MinMaxScaler
        self._scaler_args  = {} if feature_range is None else { 'feature_range': feature_range }
        self._scalers = {}

    def transform(self, X, **transform_params):
        to_scale = self._columns_to_apply(X)
        if to_scale is None:
            to_scale = self._numeric_columns(X)
        else:
            cols_ = X.select_dtypes(include=['object']).columns.values
            to_scale = [x for x in to_scale if x not in cols_]
        rest = self._rest_columns(X, to_scale).reset_index(drop=True)
        result = rest
        self._to_scale = to_scale.columns.values
        for c in self._to_scale:
            new = self._scaler_class(**self._scaler_args)
            self._scalers[c] = new
            result = pd.concat((result, pd.DataFrame(data=new.fit_transform(to_scale[[c]]), columns=[c])), axis=1)
        return result.reset_index(drop=True)

    def _do_what_i_can(self, X, method):
        cols = X.columns.values
        x_cols = set(X.columns.values)
        s_cols = set(self._scalers.keys())
        rest_cols = list(x_cols - s_cols)
        result = X[rest_cols]
        transform_cols = x_cols & s_cols
        for col in transform_cols:
            result = pd.concat((result.reset_index(drop=True),
                                pd.DataFrame(data=getattr(self._scalers[col], method)(X[[col]]), columns=[col])), axis=1)
        return result[cols]

    def inverse_transform_what_i_can(self, X):
        return self._do_what_i_can(X, 'inverse_transform')

    def transform_what_i_can(self, X):
        return self._do_what_i_can(X, 'transform')


class PcaTransformer(AbstractTransformer):
    def __init__(self, n_dim=3, prefix='pca', **other):
        super().__init__(**other)
        self.n_dim = n_dim
        self.prefix = prefix

    def transform(self, X, **transform_params):
        if X.shape[0] == 0 or X.shape[1] == 0:
            return X
        self.transformer = PCA(n_components=self.n_dim)
        to_scale = self._columns_to_apply(X)
        if to_scale is None:
            to_scale = self._numeric_columns(X)
        rest = self._rest_columns(X, to_scale)
        if to_scale.shape[0] == 0 or to_scale.shape[1] == 0:
            return X
        if self.n_dim > to_scale.shape[1]:
            return X
        new_column_names = [('%s_%s' % (self.prefix, i+1)) for i in range(self.n_dim)]
        new_data = pd.DataFrame(data=self.transformer.fit_transform(to_scale), columns=new_column_names)
        result = pd.concat((rest, new_data), axis=1)
        return result.reset_index(drop=True)


class Lagger(AbstractTransformer):
    def __init__(self, columns_strategies={}, **other):
        super().__init__(**other)
        self.columns_strategies = columns_strategies

    def _series_lag(self, df, mask, col, lag):
        s = df.loc[mask, col].tolist()
        return [s[0]]*lag + s[:-lag] if lag <= len(s) else [s[0]]*len(s)

    def _add_lag(self, X, col_name, lag, groupby=None):
        masks = [[True] * len(X)] if groupby is None else [X[groupby] == u for u in X[groupby].unique()]
        new_col_name = '%s_lag_%s' % (col_name, lag)
        X[new_col_name] = [x for b in [self._series_lag(X, m, col_name, lag) for m in masks] for x in b]

    def transform(self, X, **transform_params):
        for col_name, params in self.columns_strategies.items():
            if col_name not in X.columns.values:
                continue
            if isinstance(params, int):
                params = { 'lags': [params], 'groupby': None }
            if not isinstance(params, dict):
                params = { 'lags': params, 'groupby': None }
            if 'groupby' not in params:
                params['groupby'] = None
            [self._add_lag(X, col_name, lag, params['groupby']) for lag in params['lags']]
        return X


class CustomEncoder(AbstractTransformer):
    '''encode:
        "onehot",
        "binary",
        "backward",
        "ordinal",
        "sum",
        "poly",
        "helmert",
        "hash"
        '''
    def __init__(self, encode='onehot', **other):
        super().__init__(**other)
        self.encode = encode

    def _create_encoder(self):
        import category_encoders as ce
        _dict = {
            "onehot": ce.OneHotEncoder,
            "binary": ce.BinaryEncoder,
            "backward": ce.BackwardDifferenceEncoder,
            "ordinal": ce.OrdinalEncoder,
            "sum": ce.SumEncoder,
            "poly": ce.PolynomialEncoder,
            "helmert": ce.HelmertEncoder,
            "hash": ce.HashingEncoder,
        }
        return _dict[self.encode]

    def transform(self, X, **transform_params):
        to_transform = X.columns.values
        to_transform = list(set(to_transform) \
                            - set(self._columns_exclude) \
                            - set(X.select_dtypes(exclude=['object']).columns.values))
        to_transform = list(set(to_transform).union(self._columns_include))
        encoder_type = self._create_encoder()
        encoder = encoder_type(cols=to_transform)
        return encoder.fit_transform(X)
