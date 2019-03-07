import argparse
import itertools

from allennlp.data import Vocabulary
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from probnmn.data import JointTrainingDataset
from probnmn.data.sampler import SupervisionWeightedRandomSampler
from probnmn.models import ProgramGenerator
from probnmn.models.nmn import NeuralModuleNetwork

import probnmn.utils.checkpointing as checkpointing_utils
import probnmn.utils.common as common_utils


parser = argparse.ArgumentParser("Evaluate Joint Training.")
parser.add_argument(
    "--config-yml",
    default="configs/joint_training.yml",
    help="Path to a config file listing model and solver parameters.",
)
parser.add_argument(
    "--cpu-workers",
    type=int,
    default=0,
    help="Number of CPU workers to use for data loading."
)
# data file paths, gpu ids, checkpoint args etc.
common_utils.add_common_args(parser)
args = parser.parse_args()

# for reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # ============================================================================================
    #   INPUT ARGUMENTS AND CONFIG
    # ============================================================================================
    config = common_utils.read_config(args.config_yml)
    config = common_utils.override_config_from_opts(config, args.config_override)
    common_utils.print_config_and_args(config, args)

    device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")

    # ============================================================================================
    #   SETUP VOCABULARY, DATASET, DATALOADER, MODEL, OPTIMIZER
    # ============================================================================================
    vocabulary = Vocabulary.from_files(args.vocab_dirpath)
    val_dataset = JointTrainingDataset(
        args.tokens_val_h5, args.features_val_h5
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["optim_batch_size"],
        num_workers=args.cpu_workers
    )
    program_generator = ProgramGenerator(
        vocabulary=vocabulary,
        input_size=config["qc_model_input_size"],
        hidden_size=config["qc_model_hidden_size"],
        num_layers=config["qc_model_num_layers"],
        dropout=config["qc_model_dropout"],
    ).to(device)

    nmn = NeuralModuleNetwork(
        vocabulary=vocabulary,
        image_feature_size=tuple(config["mt_model_image_feature_size"]),
        module_channels=config["mt_model_module_channels"],
        class_projection_channels=config["mt_model_class_projection_channels"],
        classifier_linear_size=config["mt_model_classifier_linear_size"]
    ).to(device)

    program_generator.load_state_dict(torch.load(config["qc_checkpoint"])["program_generator"])
    nmn.load_state_dict(torch.load(config["mt_checkpoint"])["nmn"])

    # ============================================================================================
    #   TRAINING LOOP
    # ============================================================================================
    nmn.eval()

    from allennlp.training.metrics import BooleanAccuracy
    from scipy.stats import mode
    answer_accuracy = BooleanAccuracy()

    for i, batch in enumerate(tqdm(val_dataloader)):
        if not (i + 1) * len(batch["question"]) < args.num_val_examples:
            for key in batch:
                batch[key] = batch[key].to(device)

            batch_predictions = []

            for sample in range(1):
                # Just accumulate metrics across batches, in these models, by a forward pass.
                with torch.no_grad():
                    # sampled_programs = program_generator(
                        # batch["question"], batch["program"])["predictions"]
                    sampled_programs = program_generator(batch["question"])["predictions"]
                    output_dict = nmn(batch["image"], sampled_programs, batch["answer"])

                    batch_predictions.append(output_dict["predictions"])

            batch_predictions = torch.stack(batch_predictions, 0).cpu().numpy()

            batch_predictions = torch.tensor(mode(batch_predictions, 0)[0]).long().squeeze()
            answer_accuracy(batch_predictions, batch["answer"])

    # __pg_metrics = program_generator.get_metrics()
    # __nmn_metrics = nmn.get_metrics()

    # for metric_name in __pg_metrics:
        # print("ProgramGenerator:", metric_name, __pg_metrics[metric_name])

    # for metric_name in __nmn_metrics:
    #     print("NMN:", metric_name, __nmn_metrics[metric_name])

    print(answer_accuracy.get_metric())