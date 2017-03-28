from sklearn.ensemble import IsolationForest


class OutlierRemover(AbstractTransformer):
    def __init__(self, contamination, n_estimators=100, n_jobs=-1, **other):
        super().__init__(**other)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

    def transform(self, X, **transform_params):
        if X.shape[0] < 1/self.contamination:
            return X
        self.isolation_forest = IsolationForest(contamination=self.contamination,
                                                n_estimators=self.n_estimators,
                                                n_jobs=self.n_jobs)
        to_analyze = self._columns_to_apply(X)
        if to_analyze is None:
            to_analyze = self._numeric_columns(X)
        rest = self._rest_columns(X, to_analyze)
        self.isolation_forest.fit(to_analyze)
        labels = self.isolation_forest.predict(to_analyze)
        to_analyze['_outlier'] = labels; to_analyze = to_analyze[to_analyze['_outlier'] == 1];
        del(to_analyze['_outlier'])
        rest['_outlier'] = labels; rest = rest[rest['_outlier'] == 1]; del(rest['_outlier'])
        if self.verbose:
            print('%s Now has %s' % (self.class_name, to_analyze.shape[0]))
        return pd.concat((to_analyze, rest), axis=1)


class FillNaTransformer(AbstractTransformer):
    def __init__(self, columns_strategies, **other):
        super().__init__(**other)
        self.columns_strategies = columns_strategies

    def transform(self, X, **transform_params):
        functions = {
            'mean':        lambda c: c.fillna(value=c.mean()[0], axis=1),
            'mode':        lambda c: c.fillna(value=c.mode().iloc[0][0]
                                              if c.mode().shape[0] > 0 else c.iloc[0][0], axis=1),
            'median':      lambda c: c.fillna(value=c.median()[0], axis=1),
            'interpolate': lambda c: c.interpolate(),
            'akima':       lambda c: c.interpolate(method='akima'),
            'ffill':       lambda c: c.fillna(method='ffill'),
            'pad':         lambda c: c.fillna(method='pad'),
            'bfill':       lambda c: c.fillna(method='bfill'),
            'backfill':    lambda c: c.fillna(method='backfill'),
            'fill':        lambda c: c.fillna(method='ffill').fillna(method='bfill'),
        }
        if isinstance(self.columns_strategies, str):
            temp = {}
            [temp.update({c: {'method': self.columns_strategies}}) \
                 for c in X.columns[X.isnull().any()].tolist()]
            self.columns_strategies = temp
        for key, value in self.columns_strategies.items():
            if not isinstance(value, dict):
                value = { 'method': value }
            setter = lambda c: c.fillna(value=value['method'], axis=1)
            apply_function = functions[value['method']] if value['method'] in functions else setter
            if 'groupby' not in value:
                X[[key]] = apply_function(X[[key]])
            else:
                uniques = np.unique(X[[value['groupby']]].values)
                for i,u in enumerate(uniques):
                    m = (X[[value['groupby']]] == u).values.ravel()
                    X.loc[m, key] = apply_function(X.loc[m, key].to_frame(name=key)).values
                    if X.loc[m, key].isnull().any():
                        f = value['default'] if 'default' in value else 0
                        X.loc[m, key] = X.loc[m, key].fillna(value=f)
        return X


class UselessColumnsDropper(AbstractTransformer):
    def transform(self, X, **transform_params):
        to_process = self._columns_to_apply(X)
        to_process = X.columns.values if to_process is None else to_process.columns.values
        all_cols = list(X.columns.values)
        shitty = [c for c in to_process if len(X[c].unique()) == 1 and pd.notnull(X[c].unique()[0])]
        result = [c for c in all_cols if c not in shitty]
        return X[result]

