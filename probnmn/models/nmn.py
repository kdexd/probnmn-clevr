from typing import Dict, Tuple, Type

from allennlp.data import Vocabulary
from allennlp.training.metrics import Average, BooleanAccuracy
import torch
from torch import nn

from probnmn.modules.nmn_modules import (
    AndModule,
    AttentionModule,
    ComparisonModule,
    OrModule,
    QueryModule,
    RelateModule,
    SameModule,
)


class NeuralModuleNetwork(nn.Module):
    r"""
    A :class:`NeuralModuleNetwork` holds neural modules, a stem network, and a classifier network.
    It hooks these all together to answer a question given some scene and a program describing how
    to arrange the neural modules.

    Parameters
    ----------
    vocabulary: allennlp.data.vocabulary.Vocabulary
        AllenNLP's vocabulary. This vocabulary has three namespaces - "questions", "programs" and
        "answers", which contain respective token to integer mappings.
    image_feature_size: tuple (K, R, C), optional (default = (1024, 14, 14))
        Shape of input image features, in the form (channel, height, width).
    module_channels: int, optional (default = 128)
        Number of channels for each neural module's convolutional blocks.
    class_projection_channels: int, optional (default = 512)
        Number of channels in projected final feature map (input to classifier).
    classifier_linear_size: int, optional (default = 1024)
        Size of input to the linear classifier.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        image_feature_size: Tuple[int, int, int] = (1024, 14, 14),
        module_channels: int = 128,
        class_projection_channels: int = 1024,
        classifier_linear_size: int = 1024,
    ):
        super().__init__()
        self.vocabulary = vocabulary

        # Short-hand notations for convenience.
        __channels, __height, __width = image_feature_size

        # Exclude "@@UNKNOWN@@" answer token, our network will never generate this output through
        # regular forward pass. We set answer output as "@@UNKNOWN@@" when sampled programs are
        # invalid. __num_answers will be 28 for all practical purposes.
        __num_answers = len(vocabulary.get_index_to_token_vocabulary(namespace="answers")) - 1

        # The stem takes features from ResNet (or another feature extractor) and projects down to
        # a lower-dimensional space for sending through the Neural Module Network.
        self._stem = nn.Sequential(
            nn.Conv2d(image_feature_size[0], module_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(module_channels, module_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # The classifier takes output of the last module (which will be a Query or Equal module)
        # and produces a distribution over answers.
        self._classifier = nn.Sequential(
            nn.Conv2d(module_channels, class_projection_channels, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(class_projection_channels * __height * __width // 4, classifier_linear_size),
            nn.ReLU(),
            nn.Linear(classifier_linear_size, __num_answers),  # note no softmax here
        )

        # Instantiate a module for each program token in our vocabulary.
        self._function_modules: Dict[str, Type[nn.Module]] = {}
        for program_token in vocabulary.get_token_to_index_vocabulary("programs"):

            # We don"t need modules for the placeholders.
            if program_token in ["@@PADDING@@", "@@UNKNOWN@@", "@start@", "@end@", "unique"]:
                continue

            # Figure out which module we want we use.
            if program_token == "scene":
                # "scene" is just a flag that indicates the start of a new line of reasoning
                # we set `module` to `None` because we still need the flag "scene" in forward()
                module = None
            elif program_token == "intersect":
                module = AndModule()
            elif program_token == "union":
                module = OrModule()
            elif "equal" in program_token or program_token in {"less_than", "greater_than"}:
                module = ComparisonModule(module_channels)
            elif "query" in program_token or program_token in {"exist", "count"}:
                module = QueryModule(module_channels)
            elif "relate" in program_token:
                module = RelateModule(module_channels)
            elif "same" in program_token:
                module = SameModule(module_channels)
            else:
                module = AttentionModule(module_channels)

            # Add the module to our dictionary and register its parameters so it can learn
            self._function_modules[program_token] = module  # type: ignore
            self.add_module(program_token, module)

        # Cross Entropy Loss for answer classification.
        self._loss = nn.CrossEntropyLoss(reduction="none")

        # Record accuracy while training and validation.
        self._answer_accuracy = BooleanAccuracy()

        # Record average number of invalid programs per batch.
        self._average_invalid_programs = Average()

    def forward(self, features: torch.Tensor, programs: torch.Tensor, answers: torch.Tensor):
        r"""
        Given image features and program sequences, lay out a modular network and pass through
        the image features, further take the final feature representation output from modular
        network and pass it throuh the classifier to get the answer distribution.

        Notes
        -----
        The structure of modular network is different for each program sequence, so we just
        loop through all programs of a batch and do forward pass for each example in the loop.

        Parameters
        ----------
        features: torch.Tensor
            Input image features of shape (batch, channels, height, width).
        programs: torch.Tensor
            Program sequences padded up to maximum length, shape (batch_size, max_program_length).
        answers: torch.Tensor
            Target answers for corresponding images and programs, shape (batch_size, ).

        Returns
        -------
        Dict[str, Any]
            Model predictions, answer cross-entropy loss and (if training, ) batch metrics. A dict
            with structure::

                {
                    "predictions": torch.Tensor (shape: (batch_size, )),
                    "loss": torch.Tensor (shape: (batch_size, )),
                    "metrics": {
                        "answer_accuracy": float,
                        "average_invalid": float,
                    }
                }
        """

        # Forward all the features through the stem at once.
        # shape: (batch_size, module_channels, __height, __width)
        feat_input_volume = self._stem(features)
        batch_size, module_channels, height, width = feat_input_volume.size()

        # We compose each module network individually since they are constructed on a per-question
        # basis. Here we go through each program in the batch, construct a modular network based on
        # it, and send the image forward through the modular structure. We keep the output of the
        # last module for each program in final_module_outputs. These will be stacked to a batched
        # tensor are needed to then compute a distribution over answers for all the questions.
        final_module_outputs = []

        # Only execute the valid programs, else just assign highest possible negative logprobs to
        # answers for that example. There are 28 unique answers (excluding @@UNKNOWN@@)
        # => - ln (28) ~= 3.33
        valid_examples_mask = []
        for n in range(batch_size):
            feat_input = feat_input_volume[n].unsqueeze(0)
            output = feat_input
            saved_output = None

            try:
                for i in reversed(programs[n].cpu().numpy()):
                    module_type = self.vocabulary.get_token_from_index(i, namespace="programs")

                    # No-ops for our model - no module exists for these program tokens.
                    if module_type in {"@@PADDING@@", "@start@", "@end@", "@@UNKNOWN@@", "unique"}:
                        continue

                    module = self._function_modules[module_type]
                    if module_type == "scene":
                        # "scene" is just a flag, performing no computation, it signals to store
                        # the previous output, wil be needed later.
                        saved_output = output
                        # shape: (1, 1, __height, __width)
                        output = torch.ones_like(feat_input)[:, :1, :, :]
                        continue

                    if "equal" in module_type or module_type in {
                        "intersect",
                        "union",
                        "less_than",
                        "greater_than",
                    }:
                        # These modules take two feature maps.
                        output = module(output, saved_output)
                    else:
                        # These modules take extracted image features and a previous attention.
                        output = module(feat_input, output)

                if output.size(1) != module_channels:
                    raise ValueError("Program must end by 'encoding' as output, not 'attention'.")
                final_module_outputs.append(output)
                valid_examples_mask.append(1)
            except:  # noqa: E722
                output = torch.zeros_like(feat_input)
                final_module_outputs.append(output)
                valid_examples_mask.append(0)

        # shape: (batch_size, module_channels, __height, __width)
        final_module_outputs = torch.cat(final_module_outputs, 0)

        # shape: (batch_size, __num_answers)
        answer_logits = self._classifier(final_module_outputs)
        _, answer_predictions = torch.max(answer_logits, dim=1)

        # Replace answers of examples with invalid programs as @@UNKNOWN@@.
        valid_examples_mask_tensor = torch.tensor(valid_examples_mask)
        answer_predictions[valid_examples_mask_tensor == 0] = self.vocabulary.get_token_index(
            "@@UNKNOWN@@", namespace="answers"
        )

        # shape: (batch_size, )
        loss = self._loss(answer_logits, answers)
        # Replace loss values of examples with invalid programs as ln (__num_answers).
        loss[valid_examples_mask_tensor == 0] = 3.33

        self._answer_accuracy(answer_predictions, answers)
        self._average_invalid_programs((1 - valid_examples_mask_tensor).sum())

        # Report batch metrics only during training (training is generally slow, so this helps).
        output_dict = {"predictions": answer_predictions, "loss": loss}
        if self.training:
            output_dict["metrics"] = self.get_metrics(reset=True)
        return output_dict

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        r"""
        Return recorded answer accuracy and average invalid programs per batch.

        Parameters
        ----------
        reset: bool, optional (default = True)
            Whether to reset the accumulated metrics after retrieving them.

        Returns
        -------
        Dict[str, float]
            A dictionary with metrics ``{"answer_accuracy", "average_invalid"}``.
        """

        all_metrics: Dict[str, float] = {
            "answer_accuracy": self._answer_accuracy.get_metric(reset=reset),
            "average_invalid": self._average_invalid_programs.get_metric(reset=reset),
        }
        return all_metrics


class Flatten(nn.Module):
    r"""A PyTorch module to flatten any tensor, preserving batch dimensions."""

    def forward(self, x):
        return x.view(x.size(0), -1)
