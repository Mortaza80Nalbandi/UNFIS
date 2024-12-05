
import torch
import numpy as np
import copy
from sklearn.exceptions import NotFittedError

class _FastTensorDataLoader():
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, tensors, batch_size=32, shuffle=False):
        """
        Initialize a _FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A _FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.dataset = tensors

        self.dataset_len = self.dataset[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i + self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices)
                          for t in self.dataset)
        else:
            batch = tuple(t[self.i:self.i + self.batch_size]
                          for t in self.dataset)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class StandardScaler(object):
    def __init__(self):
        self.fitted = False

    def fit(self, train_dl: _FastTensorDataLoader):
        self.S_mean, self.X_mean, self.y_mean = [
            data.mean(axis=0) for data in train_dl.dataset]
        self.S_std, self.X_std, self.y_std = [
            data.std(axis=0) for data in train_dl.dataset]
        self.fitted = True

    def transform(self, dataloader: _FastTensorDataLoader) -> _FastTensorDataLoader:
        if not self.fitted:
            raise NotFittedError(
                'Error: The StandardScaler instance is not yet fitted. Call "fit" with appropriate arguments before using this estimator.')

        transformed_dataloder = copy.deepcopy(dataloader)

        transformed_dataloder.dataset[0] = (
            transformed_dataloder.dataset[0] - self.S_mean) / self.S_std
        transformed_dataloder.dataset[1] = (
            transformed_dataloder.dataset[1] - self.X_mean) / self.X_std

        # correct for zero division
        transformed_dataloder.dataset[1][transformed_dataloder.dataset[1].isnan(
        )] = 0.0

        # if real output required
        if len(transformed_dataloder.dataset) == 3:
            transformed_dataloder.dataset[2] = (
                transformed_dataloder.dataset[2] - self.y_mean) / self.y_std

        return transformed_dataloder

    def transform_X(self, X, inverse=False):
        if inverse:
            return X * self.X_std + self.X_mean
        return (X - self.X_mean) / self.X_std

    def transform_S(self, S, inverse=False):
        if inverse:
            return S * self.S_std + self.S_mean
        return (S - self.S_mean) / self.S_std

    def transform_y(self, y, inverse=False):
        if inverse:
            return y * self.y_std + self.y_mean

        return (y - self.y_mean) / self.y_std

    def fit_transform(self, dataloader: _FastTensorDataLoader):
        self.fit(dataloader)
        return self.transform(dataloader)


class NoneScaler(object):
    """
    This class does nothing.
    """

    def fit(self, train_dl: _FastTensorDataLoader):
        pass

    def transform(self, dataloader: _FastTensorDataLoader):
        return dataloader

    def transform_X(self, X, inverse=False):
        return X

    def transform_S(self, S, inverse=False):
        return S

    def transform_y(self, y, inverse=False):
        return y

    def fit_transform(self, dataloader: _FastTensorDataLoader):
        self.fit(dataloader)
        return self.transform(dataloader)


class DataScaler(object):

    """Class to perform data scaling operations
    The scaling technique is defined by the ``scaler`` parameter which takes one of the 
    following values: 
    - ``'Std'`` for standarizing the data to follow a normal distribution. 
    - ``'None'`` No transformation at all. 

    ----------
    normalize : str
        Type of scaling to be performed. Possible values are ``'Std'`` or  ``None``.
    """

    def __init__(self, scaler: str = 'Std'):

        if scaler == 'Std':
            self.scaler = StandardScaler()
        elif scaler == 'None':
            self.scaler = NoneScaler()
        else:
            raise ValueError(
                f"Scaler can normalize via 'Std' or 'None', but {scaler} was given.")

    def fit_transform(self, train_dl: _FastTensorDataLoader):
        """Method that estimates an scaler object using the data in ``dataset`` and scales the data in  ``dataset``

        """

        return self.scaler.fit_transform(train_dl)

    def transform(self, dataloader: _FastTensorDataLoader) -> _FastTensorDataLoader:
        """Method that scales the data in ``dataloader``
        """
        # store information from the data for plotting purposes
        # see plotmfs() from sanfis class
        self.lower_s = [np.percentile(s, 5) for s in dataloader.dataset[0].T]
        self.higher_s = [np.percentile(s, 95) for s in dataloader.dataset[0].T]
        # self.max_s =
        return self.scaler.transform(dataloader)

    def transform_X(self, X, inverse: bool = False):
        return self.scaler.transform_X(X, inverse)

    def transform_S(self, S, inverse: bool = False):
        return self.scaler.transform_S(S, inverse)

    def transform_y(self, y, inverse: bool = False):
        return self.scaler.transform_y(y, inverse)
