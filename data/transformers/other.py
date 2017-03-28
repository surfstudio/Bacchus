

class Transformer(AbstractTransformer):
    def __init__(self, f, **other):
        super().__init__(**other)
        self.f = f

    def transform(self, X, **transform_params):
        return self.f(X)


class SideEffectTransformer(AbstractTransformer):
    def __init__(self, f, **other):
        super().__init__(**other)
        self.f = f

    def transform(self, X, **transform_params):
        self.f(X)
        return X


class ColumnSelector(AbstractTransformer):
    """ by_type: object, numeric
        by_name: ['PREFIX_*'] - all columns starts with 'PREFIX_'
                 [ 'a', 'b' ] - columns named 'a' or 'b'
    """
    def __init__(self, by_type=None, by_name=None, **other):
        assert (by_type is not None) or (by_name is not None), 'Should specify by_name or by_type parameter'
        super().__init__(**other)
        self.by_type = by_type
        self.by_name = by_name

    def _select_by_type(self, X):
        to_transform = (X.select_dtypes(include=['object']) \
                        if self.by_type == 'object' else \
                        X.select_dtypes(exclude=['object'])).columns.values
        to_transform = list(set(to_transform) - set(self._columns_exclude))
        to_transform = list(set(to_transform).union(set(self._columns_include)))
        return to_transform

    def _select_by_name(self, X):
        starred = [s.replace('*', '.*') for s in [x for x in self.by_name if '*' in x]]
        exact   = [x for x in self.by_name if '*' not in x]
        exact_cols = [x for x in X.columns.values if x in exact]
        regex_cols = [x for x in X.columns.values for s in starred if len(re.findall(s, x)) > 0]
        return exact_cols + regex_cols

    def transform(self, X, **transform_params):
        cols = None
        if self.by_name is not None:
            cols = self._select_by_name(X)
        if self.by_type is not None:
            cols2 = self._select_by_type(X)
            return X[[c for c in cols if c in cols2]] if cols is not None else X[cols2]
        return X[cols]
