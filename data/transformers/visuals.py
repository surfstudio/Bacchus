

class Sorter(AbstractTransformer):
    def __init__(self, reset_index=True, **other):
        super().__init__(**other)
        self.reset_index = reset_index

    def transform(self, X, **transform_params):
        to_sort = self._columns_to_apply(X)
        if to_sort is None:
            to_sort = X.columns.values
        else:
            to_sort = to_sort.columns.values
        result = X.sort_values(by=list(to_sort))
        if self.reset_index:
            return result.reset_index(drop=True)
        return result


class ColumnsSorter(AbstractTransformer):
    def transform(self, X, **transform_params):
        exception = self._columns_include
        sortd = sorted(list(set(X.columns.values).difference(set(exception))))
        result = exception + sortd
        return X[[c for c in result]]


class RandomSampler(AbstractTransformer):
    def __init__(self, fraction=None, count=None, replace=False, **other):
        assert(fraction is not None or count is not None), \
            'Must specify "fraction" or "count" parameter'
        assert(not (fraction is not None and count is not None)), "Can't set both parameters"
        super().__init__(**other)
        self.fraction = fraction
        self.count = count
        self.replace = replace

    def transform(self, X, **transform_params):
        if self.count is None:
            self.fraction = np.clip(self.fraction, 0.0, 1.0)
            self.count = X.shape[0] * self.fraction
        else:
            self.count = np.clip(self.count, 1, X.shape[0])
        indexes = np.random.choice(range(X.shape[0]), int(self.count), replace=self.replace)
        return X.iloc[indexes]
