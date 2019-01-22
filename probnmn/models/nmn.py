from typing import Dict, Optional, Tuple

from allennlp.data import Vocabulary
from allennlp.training.metrics import Average, BooleanAccuracy
import torch
from torch import nn
from torch.nn import functional as F

from probnmn.modules import AndModule, AttentionModule, ComparisonModule, OrModule, QueryModule, \
                            RelateModule, SameModule


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NeuralModuleNetwork(nn.Module):
    """
    A ``NeuralModuleNetwork`` holds neural modules, a stem network, and a classifier network. It
    hooks these all together to answer a question given some scene and a program describing how to
    arrange the neural modules.

    Parameters
    ----------
    vocabulary: Vocabulary
        AllenNLP's vocabulary object. This vocabulary has three namespaces - "questions",
        "programs" and "answers", which contain respective token to integer mappings.
    image_feature_size: tuple (K, R, C), optional
        The shape of input feature tensors, excluding the batch size.
    module_channels: int, optional (default = 128)
        The depth of each neural module's convolutional blocks.
    class_projection_channels: int, optional (default = 512)
        The depth to project the final feature map to before classification.
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
        # regular forward pass. We hard set this when programs are invalid.
        # __num_answers will be 28 for all practical purposes.
        __num_answers = len(vocabulary.get_index_to_token_vocabulary(namespace="answers")) - 1

        # The stem takes features from ResNet (or another feature extractor) and projects down to
        # a lower-dimensional space for sending through the TbD-net
        self.stem = nn.Sequential(
            nn.Conv2d(image_feature_size[0], module_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(module_channels, module_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # The classifier takes the output of the last module (which will be a Query or Equal module)
        # and produces a distribution over answers
        self.classifier = nn.Sequential(
            nn.Conv2d(module_channels, class_projection_channels, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(class_projection_channels * __height * __width // 4, classifier_linear_size),
            nn.ReLU(),
            nn.Linear(classifier_linear_size, __num_answers),  # note no softmax here
        )

        # Instantiate a module for each program token in our vocabulary.
        self._function_modules = {}  # holds our modules
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
            self._function_modules[program_token] = module
            self.add_module(program_token, module)

        # Record accuracy while training and validation.
        self._answer_accuracy = BooleanAccuracy()
        # Record average number of invalid programs per batch.
        self._average_invalid_programs = Average()

    def forward(
        self, features: torch.Tensor, programs: torch.LongTensor, answers: torch.LongTensor
    ):

        # Forward all the features through the stem at once.
        # shape: (batch_size, module_channels, __height, __width)
        feat_input_volume = self.stem(features)
        batch_size, module_channels, height, width = feat_input_volume.size()

        # We compose each module network individually since they are constructed on a per-question
        # basis. Here we go through each program in the batch, construct a modular network based on
        # it, and send the image forward through the modular structure. We keep the output of the
        # last module for each program in final_module_outputs. These are needed to then compute a
        # distribution over answers for all the questions as a batch.
        final_module_outputs = []

        # Only execute the valid programs, else just assign highest possible negative logprobs to
        # answers for that example. There are 28 unique answers (excluding @@UNKNOWN@@)
        # => - ln (28) ~= 3.33
        valid_examples_mask = []
        for n in range(batch_size):
            feat_input = feat_input_volume[n : n + 1]
            output = feat_input
            saved_output = None

            try:
                for i in reversed(programs[n].cpu().numpy()):
                    module_type = self.vocabulary.get_token_from_index(i, namespace="programs")
                    if module_type in {"@@PADDING@@", "@start@", "@end@", "@@UNKNOWN@@", "unique"}:
                        continue  # the above are no-ops in our model

                    module = self._function_modules[module_type]
                    if module_type == "scene":
                        # Store the previous output; it will be needed later
                        # scene is just a flag, performing no computation
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
                        # These modules take two feature maps
                        output = module(output, saved_output)
                    else:
                        # These modules take extracted image features and a previous attention
                        output = module(feat_input, output)

                if output.size(1) != module_channels:
                    raise ValueError("Program must end by 'encoding' as output, not 'attention'.")
                final_module_outputs.append(output)
                valid_examples_mask.append(1)
            except:
                output = torch.zeros_like(feat_input)
                final_module_outputs.append(output)
                valid_examples_mask.append(0)

        # shape: (batch_size, module_channels, __height, __width)
        final_module_outputs = torch.cat(final_module_outputs, 0)

        # shape: (batch_size, __num_answers)
        answer_logits = self.classifier(final_module_outputs)
        _, answer_predictions = torch.max(answer_logits, dim=1)

        # Replace answers of examples with invalid programs as @@UNKNOWN@@
        valid_examples_mask = torch.tensor(valid_examples_mask)
        answer_predictions[valid_examples_mask == 0] = self.vocabulary.get_token_index(
            "@@UNKNOWN@@", namespace="answers"
        )

        loss = self._get_loss(answer_logits, answers)
        # Replace loss values of examples with invalid programs as ln (28)
        loss[valid_examples_mask == 0] = 3.33

        output_dict = {"predictions": answer_predictions, "loss": loss}
        self._answer_accuracy(answer_predictions, answers)
        self._average_invalid_programs((1 - valid_examples_mask).sum())
        if self.training:
            # Report batch answer accuracy only during training.
            output_dict["answer_accuracy"] = self._answer_accuracy.get_metric(reset=True)
            output_dict["average_invalid"] = self._average_invalid_programs.get_metric(reset=True)
        return output_dict

    @staticmethod
    def _get_loss(answer_logits: torch.Tensor, answer_targets: torch.Tensor) -> torch.Tensor:
        batch_size = answer_targets.size(0)
        # shape: (batch_size, __num_answers)
        negative_logprobs = -F.log_softmax(answer_logits, dim=-1)
        # shape: (batch_size, )
        negative_target_logprobs = negative_logprobs[torch.arange(batch_size), answer_targets]
        return negative_target_logprobs

    def get_metrics(self) -> Dict[str, float]:
        """Return recorded answer accuracy."""
        all_metrics: Dict[str, float] = {}
        if not self.training:
            all_metrics.update(
                {
                    "answer_accuracy": self._answer_accuracy.get_metric(reset=True),
                    "average_invalid": self._average_invalid_programs.get_metric(reset=True),
                }
            )
        return all_metrics
