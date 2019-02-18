"""A collection of useful methods for logging stuff to tensorboard or stdout."""
from typing import Any, Dict

from allennlp.data import Vocabulary
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import Dataset


def log_question_length_histogram(question_coding_dataset: Dataset,
                                  summary_writer: SummaryWriter):
    """
    For question coding, log a histogram of question lengths for examples with (GT) program
    supervision. This would not work for val dataset, because there's no notion of supervision
    during validation.
    """

    # Hold a list of question lengths, i-th element is length of i-th question in CLEVR v1.0
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
    question_lengths = torch.tensor(question_mask.sum(-1)).long()

    supervision_list = question_coding_dataset.get_supervision_list()

    # Retain question lengths for examples with (GT) program supervision.
    question_lengths_relevant = question_lengths[supervision_list.nonzero()]
    question_lengths_relevant = question_lengths_relevant.squeeze().cpu().numpy()

    summary_writer.add_histogram("question_lengths", question_lengths_relevant, bins="auto")


def print_question_coding_examples(batch: Dict[str, Any],
                                   iteration_output_dict: Dict[str, Any],
                                   vocabulary: Vocabulary,
                                   num: int = 10):
    for j in range(min(len(batch["question"]), num)):
        print("PROGRAM: " + " ".join(
            [vocabulary.get_token_from_index(p_index.item(), "programs")
             for p_index in batch["program"][j] if p_index != 0]
        ))

        print("SAMPLED PROGRAM: " + " ".join(
            [vocabulary.get_token_from_index(p_index.item(), "programs")
             for p_index in iteration_output_dict["predictions"]["__pg"][j]
             if p_index != 0]
        ))

        print("QUESTION: " + " ".join(
            [vocabulary.get_token_from_index(q_index.item(), "questions")
             for q_index in batch["question"][j] if q_index != 0]
        ))

        print("RECONST QUESTION: " + " ".join(
            [vocabulary.get_token_from_index(q_index.item(), "questions")
             for q_index in iteration_output_dict["predictions"]["__qr"][j]
             if q_index != 0]
        ))
        print("- " * 30)

