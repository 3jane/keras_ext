import numbers
import numpy as np
from tensorflow.keras.utils import Sequence

from typing import Optional, Any

class TimeseriesGenerator(Sequence):
    """
    Class for generating batches of temporal data.

    This class takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    stride, length of history, etc., to produce batches for
    training/validation.

    Parameters
    ----------

    data : Indexable generator (such as list or Numpy array)
        Shoulf contain consecutive data points (timesteps).
        The data should be at 2D, and axis 0 is expected to be the time dimension.
    targets : Indexable generator (such as list or Numpy array)
        Targets corresponding to timesteps in `data`.
        It should have same length as `data`.
    length : int
        Length of the output sequences (in number of timesteps).
        Default is 1.
    sampling_rate : int
        Period between successive individual timesteps within sequences. For rate `r`, timesteps
        `data[i]`, `data[i-r]`, ... `data[i - length]` are used for create a sample sequence.
        Default is 1.
    stride : int
        Period between successive output sequences.
        For stride `s`, consecutive output samples would
        be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
        Default is 1.
    target_shift : int
        Number of timesteps by which target should be shifted corresponding
        to the last row in the sampled sequence.
        Default is 0.
    start_index : int
        Data points earlier than `start_index` will not be used
        in the output sequences. This is useful to reserve part of the
        data for test or validation.
        Default is 1.
    end_index : int, optional
        Data points later than `end_index` will not be used
        in the output sequences. This is useful to reserve part of the
        data for test or validation.
        Default is None.
    shuffle : bool
        Whether to shuffle output samples, or instead draw them in chronological order.
        Default is False.
    reverse : bool
        if True, timesteps in each output sample will be in reverse chronological order.
        Default is False.
    batch_size : int
        Maximum number of samples in each batch
        Default is 1.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        A (samples, target) tuple.

    Examples
    --------
    ```python
    from keras.preprocessing.sequence import TimeseriesGenerator
    import numpy as np
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])
    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2,
                                   batch_size=2)
    assert len(data_gen) == 20
    batch_0 = data_gen[0]
    x, y = batch_0
    assert np.array_equal(x,
                          np.array([[[0], [2], [4], [6], [8]],
                                    [[1], [3], [5], [7], [9]]]))
    assert np.array_equal(y,
                          np.array([[10], [11]]))
    ```
    """

    def __init__(self,
                 data: np.ndarray,
                 targets: np.ndarray,
                 length: int = 1,
                 sampling_rate: int = 1,
                 stride: int = 1,
                 target_shift: int = 0,
                 start_index: int = 0,
                 end_index: Optional[int] = None,
                 shuffle: bool = False,
                 reverse: bool = False,
                 batch_size: int = 128,
                 random_state: Any = None):

        # Validate input lengths
        if len(data) != len(targets):
            raise ValueError('Data and targets have to be' +
                             ' of same length. '
                             'Data length is {}'.format(len(data)) +
                             ' while target length is {}'.format(len(targets)))

        # Validate random_state
        if random_state is None or random_state is np.random:
            random_state = np.random.mtrand._rand
        elif isinstance(random_state, numbers.Integral):
            random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            pass
        else:
            raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                             ' instance' % random_state)

        # Validate start_index and end_index
        end_index = len(data) - 1 if end_index is None else end_index
        if start_index > end_index:
            raise ValueError('`start_index=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))

        self.data = data
        self.targets = targets
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.target_shift = target_shift
        self.start_index = start_index
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size
        self.random_state = random_state

    def __len__(self):
        n_elements = self.end_index - self.start_index + 1
        n_minus_shift_len = n_elements - self.target_shift - (self.length - 1)
        n_batches = np.ceil(n_minus_shift_len / (self.batch_size * self.stride)).astype(int)
        return n_batches

    def __getitem__(self, index: int):
        if self.shuffle:
            # Randomly sample starting indices of each sequence
            seq_starts = self.random_state.randint(
                self.start_index,
                self.end_index + 1 - self.target_shift - (self.length - 1),
                size=self.batch_size
            )
        else:
            # Calculate starting indices of each batch
            i = self.start_index + self.batch_size * self.stride * index

            # Calculate starting indices of each sequence.
            # Take into account that the maximum starting index must be small enough
            # to allow space for the following sequence.
            seq_starts = np.arange(i,
                                   min(
                                       i + self.batch_size * self.stride,
                                       self.end_index + 1 - self.target_shift - (self.length - 1)
                                   ),
                                   self.stride)

        # Sample subsets of data
        samples = np.array([self.data[start: start + self.length: self.sampling_rate]
                            for start in seq_starts])

        # Sample corresponding targets
        targets = np.array([self.targets[start + self.target_shift + (self.length - 1)]
                            for start in seq_starts])

        # Reverse the sampled sequences, but not the targets
        if self.reverse:
            return samples[:, ::-1, ...], targets

        return samples, targets
