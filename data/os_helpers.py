import pickle
import os


def save_pickle(item, filename):
    f = open(filename, 'wb')
    pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def load_pickle(filename):
    f = open(filename, 'rb')
    result = pickle.load(f)
    f.close()
    return result


def files_in(directory, full_paths=True, extensions_include=None, extensions_exclude=None):
    assert not (extensions_include is not None and extensions_exclude is not None), \
        'Cannot specify both include and exclude extensions'
    if not full_paths:
        return [f for _,_,c in os.walk(directory) for f in c if not f.endswith('.DS_Store')]
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(directory) \
              for f in filenames if not f.endswith('.DS_Store')]
    if extensions_include is not None:
        return [item for item in result if os.path.splitext(item)[1] in extensions_include]
    if extensions_exclude is not None:
        return [item for item in result if os.path.splitext(item)[1] not in extensions_exclude]
    return result


def ensure_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
