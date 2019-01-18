import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class SupervisionWeightedRandomSampler(WeightedRandomSampler):
    """
    A ``WeightedRandomSampler`` to form a mini-batch with nearly equal number of examples
    with/without program supervision during question coding.
    """

    def __init__(self, question_coding_dataset: Dataset):

        supervision_list = question_coding_dataset.get_supervision_list()
        num_supervision = torch.sum(supervision_list)
        num_no_supervision = len(question_coding_dataset) - num_supervision

        # Set weights of indices for weighted random sampler.
        weights = torch.zeros_like(supervision_list)
        weights[supervision_list == 1] = 1 / num_supervision_yes
        weights[supervision_list == 0] = 1 / num_supervision_no

        super().__init__(
            weights=weights,
            num_samples=len(question_coding_dataset),
            replacement=True,
        )
