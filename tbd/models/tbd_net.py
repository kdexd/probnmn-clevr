from typing import Tuple

from allennlp.data import Vocabulary
import torch
from torch import nn

from tbd.modules import AndModule, AttentionModule, ComparisonModule, OrModule, QueryModule, \
                        RelateModule, SameModule


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class TbDNet(nn.Module):
    """ The real deal. A full Transparency by Design network (TbD-net).

    Extended Summary
    ----------------
    A :class:`TbDNet` holds neural :mod:`modules`, a stem network, and a classifier network. It
    hooks these all together to answer a question given some scene and a program describing how to
    arrange the neural modules.

    Parameters
    ----------
    vocabulary: Vocabulary
        AllenNLP's vocabulary object. This vocabulary has three namespaces - "questions",
        "programs" and "answers", which contain respective token to integer mappings.
    image_feature_size : the tuple (K, R, C), optional
        The shape of input feature tensors, excluding the batch size.
    module_channels : int, optional (default = 128)
        The depth of each neural module's convolutional blocks.
    class_projection_channels : int, optional (default = 512)
        The depth to project the final feature map to before classification.
    """

    def __init__(self,
                 vocabulary: Vocabulary,
                 image_feature_size: Tuple[int, int, int] = (1024, 14, 14),
                 module_channels: int = 128,
                 class_projection_channels: int = 1024,
                 classifier_linear_size: int = 1024):
        super().__init__()
        self.vocabulary = vocabulary

        # Short-hand notations for convenience.
        __channels, __height, __width = image_feature_size
        __num_answers = len(vocabulary.get_index_to_token_vocabulary(namespace="answers"))

        # The stem takes features from ResNet (or another feature extractor) and projects down to
        # a lower-dimensional space for sending through the TbD-net
        self.stem = nn.Sequential(
            nn.Conv2d(image_feature_size[0], module_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(module_channels, module_channels, kernel_size=3, padding=1),
            nn.ReLU()
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
            nn.Linear(classifier_linear_size, __num_answers)  # note no softmax here
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

        # this is used as input to the first AttentionModule in each program
        ones = torch.ones(1, 1, __width, __height)
        self._ones_var = ones.cuda() if torch.cuda.is_available() else ones

        self._attention_sum = 0

    @property
    def attention_sum(self):
        """
        Returns
        -------
        attention_sum : int
            The sum of attention masks produced during the previous forward pass, or zero if a
            forward pass has not yet happened.

        Extended Summary
        ----------------
        This property holds the sum of attention masks produced during a forward pass of the model.
        It will hold the sum of all the AttentionModule, RelateModule, and SameModule outputs. This
        can be used to regularize the output attention masks, hinting to the model that spurious
        activations that do not correspond to objects of interest (e.g. activations in the 
        background) should be minimized. For example, a small factor multiplied by this could be
        added to your loss function to add this type of regularization as in:

            loss = xent_loss(outs, answers)
            loss += executor.attention_sum * 2.5e-07
            loss.backward()

        where `xent_loss` is our loss function, `outs` is the output of the model, `answers` is the
        PyTorch `Tensor` containing the answers, and `executor` is this model. The above block
        will penalize the model's attention outputs multiplied by a factor of 2.5e-07 to push the
        model to produce sensible, minimal activations.
        """
        return self._attention_sum

    def forward(self, feats, programs):
        batch_size = feats.size(0)
        assert batch_size == len(programs)

        feat_input_volume = self.stem(feats)  # forward all the features through the stem at once

        # We compose each module network individually since they are constructed on a per-question
        # basis. Here we go through each program in the batch, construct a modular network based on
        # it, and send the image forward through the modular structure. We keep the output of the
        # last module for each program in final_module_outputs. These are needed to then compute a
        # distribution over answers for all the questions as a batch.
        final_module_outputs = []
        self._attention_sum = 0
        for n in range(batch_size):
            feat_input = feat_input_volume[n:n+1] 
            output = feat_input
            saved_output = None
            for i in reversed(programs.data[n].cpu().numpy()):
                module_type = self.vocab["program_idx_to_token"][i]
                if module_type in {"@@PADDING@@", "@start@", "@end@", "@@UNKNOWN@@", "unique"}:
                    continue  # the above are no-ops in our model
                
                module = self._function_modules[module_type]
                if module_type == "scene":
                    # store the previous output; it will be needed later
                    # scene is just a flag, performing no computation
                    saved_output = output
                    output = self._ones_var
                    continue
                
                if "equal" in module_type or module_type in {"intersect", "union", "less_than",
                                                             "greater_than"}:
                    output = module(output, saved_output)  # these modules take two feature maps
                else:
                    # these modules take extracted image features and a previous attention
                    output = module(feat_input, output)

                if any(t in module_type for t in ["filter", "relate", "same"]):
                    self._attention_sum += output.sum()
                    
            final_module_outputs.append(output)
            
        final_module_outputs = torch.cat(final_module_outputs, 0)
        return self.classifier(final_module_outputs)

    def forward_and_return_intermediates(self, program_var, feats_var):
        """ Forward program `program_var` and image features `feats_var` through the TbD-Net
        and return an answer and intermediate outputs.

        Parameters
        ----------
        program_var : torch.Tensor
            The program to carry out.

        feats_var : torch.Tensor
            The image features to operate on.

        Returns
        -------
        Tuple[str, List[Tuple[str, numpy.ndarray]]]
            A tuple of (answer, [(operation, attention), ...]). Note that some of the
            intermediates will be `None`, which indicates a break in the logic chain. For
            example, in the question:
                "What color is the cube to the left of the sphere and right of the cylinder?"
            We have 3 distinct chains of reasoning. We first localize the sphere and look left. We
            then localize the cylinder and look right. Thirdly, we look at the intersection of these
            two, and find the cube.
        """
        intermediaries = []
        # the logic here is the same as self.forward()
        scene_input = self.stem(feats_var)
        output = scene_input
        saved_output = None
        for i in reversed(program_var.data.cpu().numpy()[0]):
            module_type = self.vocab["program_idx_to_token"][i]
            if module_type in {"@@PADDING@@", "@start@", "@end@", "@@UNKNOWN@@", "unique"}:
                continue

            module = self._function_modules[module_type]
            if module_type == "scene":
                saved_output = output
                output = self._ones_var
                intermediaries.append(None) # indicates a break/start of a new logic chain
                continue

            if "equal" in module_type or module_type in {"intersect", "union", "less_than",
                                                         "greater_than"}:
                output = module(output, saved_output)
            else:
                output = module(scene_input, output)

            if module_type in {"intersect", "union"}:
                intermediaries.append(None) # this is the start of a new logic chain

            if module_type in {"intersect", "union"} or any(s in module_type for s in ["same",
                                                                                       "filter",
                                                                                       "relate"]):
                intermediaries.append((module_type, output.data.cpu().numpy().squeeze()))

        _, pred = self.classifier(output).max(1)
        return (self.vocab["answer_idx_to_token"][pred.item()], intermediaries)


def load_tbd_net(checkpoint, vocab):
    """ Convenience function to load a TbD-Net model from a checkpoint file.

    Parameters
    ----------
    checkpoint : Union[pathlib.Path, str]
        The path to the checkpoint.

    vocab : Dict[str, Dict[any, any]]
        The vocabulary file associated with the TbD-Net. For an extended description, see above.

    Returns
    -------
    torch.nn.Module
        The TbD-Net model.

    Notes
    -----
    This pushes the TbD-Net model to the GPU if a GPU is available.
    """
    tbd_net = TbDNet(vocab)
    tbd_net.load_state_dict(torch.load(str(checkpoint), map_location={"cuda:0": "cpu"}))
    if torch.cuda.is_available():
        tbd_net.cuda()
    return tbd_net
