import logging
from typing import Any, Dict, List, Optional, Type

from allennlp.data import Vocabulary
import torch
from torch import nn
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import QuestionCodingDataset
from ._evaluator import _Evaluator


logger: logging.Logger = logging.getLogger(__name__)


class QuestionCodingEvaluator(_Evaluator):
    def __init__(
        self,
        config: Config,
        models: Dict[str, Type[nn.Module]],
        device: torch.device,
    ):
        self._C = config

        if self._C.PHASE != "question_coding":
            raise ValueError(
                f"Trying to initialize a QuestionCodingEvaluator, expected config PHASE to be "
                f"question_coding, found {self._C.PHASE}"
            )

        # Initialize vocabulary, dataloader and model.
        self._vocabulary = Vocabulary.from_files(self._C.DATA.VOCABULARY)

        # There is no notion of "supervision" during evaluation.
        dataset = QuestionCodingDataset(self._C.DATA.VAL.TOKENS)
        dataloader = DataLoader(dataset, batch_size=self._C.OPTIM.BATCH_SIZE)

        super().__init__(config=config, dataloader=dataloader, models=models, device=device)

        # These will be a part of `self._models`, keep these handles for convenience.
        self._program_generator = self._models["program_generator"]
        self._question_reconstructor = self._models["question_reconstructor"]

    def evaluate(self, num_batches: Optional[int] = None):
        eval_metrics = super().evaluate(num_batches)

        # ----------------------------------------------------------------------------------------
        # PRINT MODEL PREDICTIONS FOR FIVE EXAMPLES (OF FIRST BATCH)
        # ----------------------------------------------------------------------------------------
        self._program_generator.eval()
        self._question_reconstructor.eval()

        for batch in self._dataloader:
            for key in batch:
                batch[key] = batch[key].to(self._device)
            break

        with torch.no_grad():
            output_dict = self._do_iteration(batch)

        print("\n")
        for j in range(5):
            program_gt_tokens: List[str] = [
                self._vocabulary.get_token_from_index(p_index.item(), "programs")
                for p_index in batch["program"][j]
                if p_index != 0
            ]
            program_sampled_tokens: List[str] = [
                self._vocabulary.get_token_from_index(p_index.item(), "programs")
                for p_index in output_dict["program_generator"]["predictions"][j]
                if p_index != 0
            ]

            question_gt_tokens: List[str] = [
                self._vocabulary.get_token_from_index(q_index.item(), "questions")
                for q_index in batch["question"][j]
                if q_index != 0
            ]
            question_reconstruction_tokens: List[str] = [
                self._vocabulary.get_token_from_index(q_index.item(), "questions")
                for q_index in output_dict["question_reconstructor"]["predictions"][j]
                if q_index != 0
            ]

            logger.info("PROGRAM: " + " ".join(program_gt_tokens))
            logger.info("SAMPLED PROGRAM: " + " ".join(program_sampled_tokens))
            logger.info("QUESTION: " + " ".join(question_gt_tokens))
            logger.info("RECONST QUESTION: " + " ".join(question_reconstruction_tokens))
            print("- " * 30)

        self._program_generator.train()
        self._question_reconstructor.train()
        # ----------------------------------------------------------------------------------------

        return eval_metrics

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one iteration, take a forward pass to accumulate metrics in model objects."""

        # Forward pass through program_generator and question reconstructor.
        # keys: {"predictions", "loss"}
        program_generator_output_dict = self._program_generator(
            batch["question"], batch["program"]
        )
        question_reconstructor_output_dict = self._question_reconstructor(
            batch["program"], batch["question"]
        )
        return {
            "program_generator": program_generator_output_dict,
            "question_reconstructor": question_reconstructor_output_dict,
        }
