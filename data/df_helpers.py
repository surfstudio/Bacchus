import itertools
import pandas as pd


def cartesian(df1, df2):
    rows = itertools.product(df1.iterrows(), df2.iterrows())
    df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
    return df.reset_index(drop=True)
