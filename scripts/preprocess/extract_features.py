r"""
Extract image features from CLEVR v1.0 images using ResNet-101 after stage-3. Features dimensions
would be (1024, 14, 14). If you wish to extract stage-4, they will be (2048, 7, 7).
"""
import argparse
import glob
import os
from typing import List

import h5py
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet101
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="Extract image features from CLEVR v1.0 images using a pre-trained CNN."
)
parser.add_argument(
    "-i",
    "--image-dir",
    default="data/images/val",
    help="Path to a directory containing CLEVR v1.0 images of a particular split.",
)
parser.add_argument(
    "-o",
    "--output-h5path",
    default="data/clevr_val_features.h5",
    help="Path to save extracted image features in an H5 file.",
)
parser.add_argument("-s", "--split", default="val")

parser.add_argument(
    "--gpu-ids", required=True, nargs="+", type=int, help="List of GPU ids to use (-1 for CPU)."
)
parser.add_argument(
    "--cpu-workers", type=int, default=0, help="Number of CPU workers to use for data loading."
)
parser.add_argument("-b", "--batch-size", type=int, default=128)


class ClevrImagesDataset(Dataset):
    r"""
    A PyTorch Dataset which returns image tensors with appropriate pre-processing required for the
    inputs to ResNet-101.

    Parameters
    ----------
    image_paths: str
        List of image paths for CLEVR images corresponding to a particular split.
    """

    # These image dimensions are required to get (1024, 14, 14) dimension features.
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224

    def __init__(self, image_paths: List[str]):
        self._image_paths = image_paths
        self._transform = Compose(
            [
                Resize((self.IMAGE_HEIGHT, self.IMAGE_WIDTH)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224]),
            ]
        )

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, index):
        image = Image.open(self._image_paths[index]).convert("RGB")
        image_tensor = self._transform(image)
        return image_tensor


def main(args: argparse.Namespace):

    # --------------------------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER AND MODEL
    # --------------------------------------------------------------------------------------------

    # Collect all image paths in the image directory.
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    dataset = ClevrImagesDataset(image_paths)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.cpu_workers, shuffle=False
    )

    # Set device according to specified GPU ids.
    device = torch.device(f"cuda:{args.gpu_ids[0]}" if args.gpu_ids[0] >= 0 else "cpu")

    # Load ResNet-101 with ImageNet pre-trained weights.
    resnet = resnet101(pretrained=True)
    feature_extractor = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
    )
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    # Enable multi-GPU execution if needed.
    if len(args.gpu_ids) > 1 and -1 not in args.gpu_ids:
        feature_extractor = nn.DataParallel(feature_extractor)

    # --------------------------------------------------------------------------------------------
    #   EXTRACT AND SAVE FEATURES
    # --------------------------------------------------------------------------------------------

    # Accumulate all the features according to sorted image ids.
    features = np.zeros((len(image_paths), 1024, 14, 14))

    counter = 0
    for batch in tqdm(dataloader):
        # shape: (batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
        batch = batch.to(device)

        with torch.no_grad():
            # shape: (batch_size, 1024, 14, 14)
            output_features = feature_extractor(batch)

        features[counter : counter + len(batch)] = output_features.cpu().numpy()
        counter += len(batch)

    output_h5 = h5py.File(args.output_h5path, "w")
    output_h5.attrs["split"] = args.split
    output_h5["features"] = features
    output_h5.close()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
