import h5py
import torch
from torch.utils.data import Dataset


class ClevrProgramsDataset(Dataset):
    # TODO: extend to full CLEVR dataset.

    """Provides programs as tokenized sequences to train the ``ProgramPrior``.

    Parameters
    ----------
    programs_hdfpath : str
        Path to HDF file containing program tokens and lengths.
    """

    def __init__(self, programs_hdfpath: str):
        with h5py.File(programs_hdfpath, "r") as programs_hdf:
            # shape: (dataset_size, max_program_length)
            self.program_tokens = programs_hdf["program_tokens"][:]
            # shape: (dataset_size, )
            self.program_lengths = programs_hdf["program_lengths"][:]

            self._split = programs_hdf.attrs["split"]
            self._style = programs_hdf.attrs["style"]

    @property
    def split(self):
        return self._split

    @property
    def style(self):
        return self._style

    def __len__(self):
        return len(self.program_lengths)

    def __getitem__(self, index):
        return {
            "program_tokens": torch.tensor(self.program_tokens[index, :]),
            "program_lengths": torch.tensor(self.program_lengths[index]).long()
        }
