import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class SupervisionWeightedRandomSampler(WeightedRandomSampler):
    """
    A ``WeightedRandomSampler`` to form a mini-batch with nearly equal number of examples
    with/without program supervision during question coding.
    """

    def __init__(self,
                 question_coding_dataset: Dataset,
                 max_question_length: int = 45):

        self._supervision_list = question_coding_dataset.get_supervision_list().float()

        # Hold a list of question lengths, element is length of i-th question in CLEVR v1.0
        # train split, excluding "@start@" and "@end@".
        tokens = question_coding_dataset._tokens

        if isinstance(tokens, list):
            # In Overfit mode.
            questions = [torch.tensor(token["question"]) for token in tokens]
            questions = torch.stack(questions)
            question_mask = questions != 0
        else:
            question_mask = tokens.questions[:] != 0

        # Shape: (699989, ) for CLEVR v1.0 train split
        self._question_lengths = torch.tensor(question_mask.sum(-1)).long()

        supervision_mask = (self._question_lengths <= max_question_length).float()

        # Examples with questions longer than specified will be temporarily dropped.
        num_examples_in_curriculum = torch.sum(supervision_mask)
        masked_supervision_list = self._supervision_list * supervision_mask

        num_supervision = torch.sum(masked_supervision_list)
        num_no_supervision = num_examples_in_curriculum - num_supervision

        # Set weights of indices for weighted random sampler.
        weights = torch.zeros_like(masked_supervision_list)
        weights[masked_supervision_list == 1] = 1 / num_supervision
        weights[masked_supervision_list == 0] = 1 / num_no_supervision

        # Set probability of sampling masked examples as zero.
        weights[supervision_mask == 0] = 0

        super().__init__(
            weights=weights,
            num_samples=len(question_coding_dataset),
            replacement=True,
        )
