from typing import List, Optional

import torch

from allennlp.data import Vocabulary
from allennlp.training.metrics import SequenceAccuracy


class SemanticQuestionReconstructionAccuracy(SequenceAccuracy):
    r"""A metric which computes question reconstruction accuracy in a "semantic" sense, this
    considers certain synonym words equivalent, such as ``("object", "thing")``.

    The mapping of synonyms is taken from
    `CLEVR dataset generation code <https://github.com/facebookresearch/clevr-dataset-gen>`_.

    Parameters
    ----------
    vocabulary: allennlp.data.vocabulary.Vocabulary
        AllenNLP's vocabulary. This vocabulary has three namespaces - "questions", "programs" and
        "answers", which contain respective token to integer mappings.

    """

    SYNONYM_TUPLES = [
        ("on the left side of", "left"),
        ("to the left of", "left"),
        ("left of", "left"),
        ("on the right side of", "right"),
        ("to the right of", "right"),
        ("right of", "right"),
        ("in front of", "front"),
        ("object", "thing"),
        ("ball", "sphere"),
        ("block", "cube"),
        ("big", "large"),
        ("tiny", "small"),
        ("shiny", "metal"),
        ("metallic", "metal"),
        ("matte", "rubber"),
    ]

    def __init__(self, vocabulary: Vocabulary):
        super().__init__()
        self._vocabulary = vocabulary

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_questions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):

        # Remove beam dimension temporarily.
        # shape: (batch_size, max_question_length)
        predictions = predictions.squeeze(1)
        batch_size, max_question_length = predictions.size()

        # Just convert predictions and gold questions as strings and replace synonyms.
        predictions_tokens: List[List[str]] = [
            [
                self._vocabulary.get_token_from_index(q.item(), namespace="questions")
                for q in question
            ]
            for question in predictions
        ]

        gold_questions_tokens: List[List[str]] = [
            [
                self._vocabulary.get_token_from_index(q.item(), namespace="questions")
                for q in question
            ]
            for question in gold_questions
        ]

        for i, prediction_tokens in enumerate(predictions_tokens):
            prediction_str: str = " ".join(prediction_tokens)
            for synonym_tuple in self.SYNONYM_TUPLES:
                prediction_str = prediction_str.replace(synonym_tuple[0], synonym_tuple[1])
            predictions_tokens[i] = prediction_str.split(" ")

            # Retain maximum length by appending padding tokens (replacing multi-word synonyms
            # might reduct eh question length.)
            if len(predictions_tokens[i]) < max_question_length:
                predictions_tokens[i].extend(
                    ["@@PADDING@@"] * (max_question_length - len(predictions_tokens[i]))
                )

        for i, gold_question_tokens in enumerate(gold_questions_tokens):
            gold_question_str: str = " ".join(gold_question_tokens)
            for synonym_tuple in self.SYNONYM_TUPLES:
                gold_question_str = gold_question_str.replace(synonym_tuple[0], synonym_tuple[1])
            gold_questions_tokens[i] = gold_question_str.split(" ")

            # Retain maximum length by appending padding tokens (replacing multi-word synonyms
            # might reduct eh question length.)
            if len(gold_questions_tokens[i]) < max_question_length:
                gold_questions_tokens[i].extend(
                    ["@@PADDING@@"] * (max_question_length - len(gold_questions_tokens[i]))
                )

        # Convert the predictions and gold tokens back to tensors.
        predictions_indices: List[List[int]] = [
            [self._vocabulary.get_token_index(q, namespace="questions") for q in question]
            for question in predictions_tokens
        ]
        gold_questions_indices: List[List[int]] = [
            [self._vocabulary.get_token_index(q, namespace="questions") for q in question]
            for question in gold_questions_tokens
        ]

        predictions = torch.tensor(predictions_indices).long().to(predictions.device)
        gold_questions = torch.tensor(gold_questions_indices).long().to(gold_questions.device)

        # Add beam dimension because AllenNLP's sequence accuracy expects this.
        # shape: (batch_size, 1, max_question_length)
        predictions = predictions.unsqueeze(1)

        super().__call__(predictions, gold_questions, mask)
