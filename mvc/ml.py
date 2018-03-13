import numpy as np


def get_X_and_y(d, test_col_str, other_col_to_keep, remove_nans=False):
    y = np.nanmax(d[test_col_str], axis=1)

    col_names = other_col_to_keep.tolist() + test_col_str.tolist()
    X = d[col_names].values

    nan_row_idx = np.isnan(X).any(axis=1)
    if remove_nans:
        X = X[~nan_row_idx, :]
        y = y[~nan_row_idx]
        print(f'Removed {np.sum(nan_row_idx)} rows')
    else:
        if np.any(nan_row_idx):
            print(
                f'Warning: {np.sum(nan_row_idx)} rows have nans. You should remove it or use Imputer'
            )
    return X, y, col_names, nan_row_idx


class Normalize:
    def __init__(self, to_normalize=None, ref='max'):
        self.to_normalize = to_normalize
        self.ref = ref

    def transform(self, X, y):
        normalize = lambda a, b: a * 100 / b
        X_out, y_out = [], []
        if X.any():
            X_out = X.copy()
            if self.ref is 'max':
                self.ref_vector = np.nanmax(X_out[:, self.to_normalize], axis=1)
            else:
                self.ref_vector = X_out[:, self.ref].ravel()
            X_out[:, self.to_normalize] = np.apply_along_axis(
                normalize, 0, X_out[:, self.to_normalize], self.ref_vector)
            if y.any():
                y_out = np.apply_along_axis(normalize, 0, y, self.ref_vector)
        return X_out, y_out

    def inverse_transform(self, X, y):
        denormalize = lambda a, b: a / 100 * b
        X_out, y_out = [], []
        if X.any():
            X_out = X.copy()
            X_out[:, self.to_normalize] = np.apply_along_axis(
                denormalize, 0, X_out[:, self.to_normalize], self.ref_vector)
        if y.any():
            y_out = np.apply_along_axis(denormalize, 0, y, self.ref_vector)
        return X_out, y_out
