r"""
A Reader simply reads data from disk and returns it almost as is. Readers should be utilized by
PyTorch :class:`torch.utils.data.Dataset`. Any type of data pre-processing is not recommended in
the reader, such as tokenizing words to integers, embedding tokens, or passing an image through
a pre-trained CNN.

Each reader must implement three methods:

    - ``__len__`` to return the length of data this Reader can read.
    - ``__getitem__`` to return data based on a unique index (which behaves as a primary key).
    - ``keys`` to return a list of possible primary keys (or indices) this Reader can provide
      data of.
"""
import h5py


class ClevrTokensReader(object):
    r"""
    A Reader for retrieving tokenized CLEVR programs, questions and answers, and corresponding
    image indices from a single HDF file containing this pre-processed data.

    Parameters
    ----------
    tokens_h5path: str
        Path to an HDF file containing tokenized programs, questions, answers and corresponding
        image indices.
    """

    def __init__(self, tokens_h5path: str):
        # questions, image indices, programs, and answers are small enough to load into memory
        with h5py.File(tokens_h5path, "r") as clevr_tokens:
            self._split = clevr_tokens.attrs["split"]

            if self._split != "test":
                self.programs = clevr_tokens["programs"][:]
                self.answers = clevr_tokens["answers"][:]

            self.questions = clevr_tokens["questions"][:]
            self.image_indices = clevr_tokens["image_indices"][:]

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, index):
        if self.split == "test":
            return {
                "question": self.questions[index],
                "image_index": self.image_indices[index],
            }
        else:
            return {
                "program": self.programs[index],
                "question": self.questions[index],
                "answer": self.answers[index],
                "image_index": self.image_indices[index],
            }

    @property
    def split(self):
        return self._split


class ClevrImageFeaturesReader(object):
    r"""
    A Reader for retrieving pre-extracted image features from CLEVR images. We typically use
    features extracted using ResNet-101.

    Example of an HDF file::

        features_train.h5
        |--- "features" [shape: (num_images, channels, height, width)]
        +--- .attrs ("split", "train")

    Parameters
    ----------
    features_h5path: str
        Path to an HDF file containing a 'dataset' of pre-extracted image features.
    in_memory: bool, optional (default = True)
        Whether to load all image features in memory.
    """

    def __init__(self, features_h5path: str, in_memory: bool = True):
        self.features_h5path = features_h5path
        self._in_memory = in_memory

        # Image feature files are typically 50-100 GB in size, careful when loading in memory.
        with h5py.File(self.features_h5path, "r") as features_hdf:
            self._split = features_hdf.attrs["split"]
            if self._in_memory:
                self.features = features_hdf["features"][:]
            else:
                self.features = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        if self._in_memory:
            features = self.features[index]
        else:
            # read chunk from file everytime if not loaded in memory
            with h5py.File(self.features_h5path, "r") as features_hdf:
                features = features_hdf["features"][index]
        return features

    @property
    def split(self):
        return self._split
