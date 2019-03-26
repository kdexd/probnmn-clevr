import argparse

from allennlp.data import Vocabulary
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from probnmn.config import Config
from probnmn.data import JointTrainingDataset
from probnmn.models import ProgramGenerator, NeuralModuleNetwork
import probnmn.utils.common as common_utils


parser = argparse.ArgumentParser("Evaluate Joint Training.")
parser.add_argument(
    "--config-yml",
    default="configs/joint_training.yml",
    help="Path to a config file listing model and solver parameters.",
)
parser.add_argument(
    "--checkpoint-pthpath",
    default="data/joint_training_1000_ours_best.pth",
    help="Path to joint training pre-trained checkpoint."
)
# Data file paths, gpu ids, checkpoint args etc.
common_utils.add_common_args(parser)


if __name__ == "__main__":
    # ============================================================================================
    #   INPUT ARGUMENTS AND CONFIG
    # ============================================================================================
    _A = parser.parse_args()

    # Create a config with default values, then override from config file, and _A.
    # This config object is immutable, nothing can be changed in this anymore.
    _C = Config(_A.config_yml, _A.config_override)
    common_utils.print_config_and_args(_C, _A)

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)
    torch.cuda.manual_seed_all(_C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set device according to specified GPU ids.
    device = torch.device("cuda", _A.gpu_ids[0]) if _A.gpu_ids[0] >= 0 else torch.device("cpu")

    # ============================================================================================
    #   SETUP VOCABULARY, DATASET, DATALOADER, MODEL, OPTIMIZER
    # ============================================================================================
    vocabulary = Vocabulary.from_files(_A.vocab_dirpath)
    val_dataset = JointTrainingDataset(_A.tokens_val_h5, _A.features_val_h5)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE,
        num_workers=_A.cpu_workers
    )
    program_generator = ProgramGenerator(
        vocabulary=vocabulary,
        input_size=_C.PROGRAM_GENERATOR.INPUT_SIZE,
        hidden_size=_C.PROGRAM_GENERATOR.HIDDEN_SIZE,
        num_layers=_C.PROGRAM_GENERATOR.NUM_LAYERS,
        dropout=_C.PROGRAM_GENERATOR.DROPOUT,
    ).to(device)

    nmn = NeuralModuleNetwork(
        vocabulary=vocabulary,
        image_feature_size=tuple(_C.NMN.IMAGE_FEATURE_SIZE),
        module_channels=_C.NMN.MODULE_CHANNELS,
        class_projection_channels=_C.NMN.CLASS_PROJECTION_CHANNELS,
        classifier_linear_size=_C.NMN.CLASSIFIER_LINEAR_SIZE,
    ).to(device)

    # Load checkpoints from joint training phase.
    joint_training_checkpoint = torch.load(_A.checkpoint_pthpath)
    program_generator.load_state_dict(joint_training_checkpoint["program_generator"])
    nmn.load_state_dict(joint_training_checkpoint["nmn"])

    # ============================================================================================
    #   TRAINING LOOP
    # ============================================================================================
    nmn.eval()

    for i, batch in enumerate(tqdm(val_dataloader, desc="validation")):
        if not (i + 1) * len(batch["question"]) < _A.num_val_examples:
            for key in batch:
                batch[key] = batch[key].to(device)

            for sample in range(1):
                # Just accumulate metrics across batches, in these models, by a forward pass.
                with torch.no_grad():
                    sampled_programs = program_generator(
                        batch["question"], batch["program"])["predictions"]
                    sampled_programs = program_generator(batch["question"])["predictions"]
                    output_dict = nmn(batch["image"], sampled_programs, batch["answer"])

    pg_metrics = program_generator.get_metrics()
    for metric_name in pg_metrics:
        print("ProgramGenerator:", metric_name, pg_metrics[metric_name])

    nmn_metrics = nmn.get_metrics()
    for metric_name in nmn_metrics:
        print("NMN:", metric_name, nmn_metrics[metric_name])
