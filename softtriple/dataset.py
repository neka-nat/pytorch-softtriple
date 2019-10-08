import random
from fuel.datasets import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import BatchSizeScheme, SequentialScheme
from .random_fixed_size_crop_mod import RandomFixedSizeCrop


def get_streams(path, batch_size=64, crop_size=224):
    '''
    args:
        path (str): data file path.
        batch_size (int):
            number of examples per batch
        crop_size (int or tuple of ints):
            height and width of the cropped image.
    '''

    dataset_class = H5PYDataset
    dataset_train = dataset_class(path, ['train'], load_in_memory=True)
    dataset_test = dataset_class(path, ['test'], load_in_memory=True)

    if not isinstance(crop_size, tuple):
        crop_size = (crop_size, crop_size)

    scheme = ShuffledBatchSizeScheme(examples=dataset_train.num_examples, batch_size=batch_size)
    stream = DataStream(dataset_train, iteration_scheme=scheme)
    stream_train = RandomFixedSizeCrop(stream, which_sources=('images',),
                                       random_lr_flip=True,
                                       window_shape=crop_size)

    stream_train_eval = RandomFixedSizeCrop(DataStream(
        dataset_train, iteration_scheme=SequentialScheme(
            dataset_train.num_examples, batch_size)),
        which_sources=('images',), center_crop=True, window_shape=crop_size)
    stream_test = RandomFixedSizeCrop(DataStream(
        dataset_test, iteration_scheme=SequentialScheme(
            dataset_test.num_examples, batch_size)),
        which_sources=('images',), center_crop=True, window_shape=crop_size)

    return stream_train, stream_train_eval, stream_test


class ShuffledBatchSizeScheme(BatchSizeScheme):
    def __init__(self, examples, batch_size):
        self.examples = examples
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return self._generate_indexes()

    def _generate_indexes(self):
        return random.sample(list(range(self.examples)), self.batch_size)

    def get_request_iterator(self):
        return self