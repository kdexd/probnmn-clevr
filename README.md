Probabilistic Neural-symbolic Models
====================================

Code for our ICML 2019 paper:

**[Probabilistic Neural-Symbolic Models for Interpretable Visual Question Answering][1]**
Ramakrishna Vedantam, Karan Desai, Stefan Lee, Marcus Rohrbach, Dhruv Batra, Devi Parikh

  * [Setup and Dependencies](#setup-and-dependencies)
  * [Data Preprocessing](#data-preprocessing)

![probnmn-model](docs/_static/probnmn_model.jpg)

If you find this code useful, please consider citing:

```text
@inproceedings{vedantam2019probabilistic,
  title={Probabilistic Neural-symbolic Models for Interpretable Visual Question Answering},
  author={Ramakrishna Vedantam and Karan Desai and Stefan Lee and Marcus Rohrbach and Dhruv Batra and Devi Parikh},
  booktitle={ICML},
  year={2019}
}
```


Setup and Dependencies
----------------------

This codebase uses PyTorch v1.0 and provides out of the box support with CUDA 9 and CuDNN 7. The
recommended way to set up this codebase is throgh Anaconda / Miniconda, as a developement package:

1. Install Anaconda or Miniconda distribution based on Python3+ from their [downloads' site][2].
2. Clone this repository and create an environment:

```sh
git clone https://www.github.com/kdexd/probnmn-clevr
conda create -n probnmn python=3.6
```

3. Activate the environment and install all dependencies.

```sh
conda activate probnmn
cd probnmn-clevr/
pip install -r requirements.txt
```

4. Install this codebase as a package in development version.

```sh
python setup.py develop
```


Data Preprocessing
------------------

1. This codebase assumes all the data to be in `$PROJECT_ROOT/data` directory by default, although
   custom paths can be provided through config. Download CLEVR v1.0 dataset from [here][3] and
   symlink it as follows:

```text
$PROJECT_ROOT/data
    |—— CLEVR_test_questions.json
    |—— CLEVR_train_questions.json
    |—— CLEVR_val_questions.json
    `—— images
        |—— train
        |   |—— CLEVR_train_000000.png
        |   `—— CLEVR_train_000001.png ...
        |—— val
        |   |—— CLEVR_val_000000.png
        |   `—— CLEVR_val_000001.png ...
        `—— test
            |—— CLEVR_test_000000.png
            `—— CLEVR_test_000001.png ...
```

2. Build a vocabulary out of CLEVR programs, questions and answers, which can be read by AllenNLP,
   and will be used throughout the training and evaluation procedures. This will create a directory
   with separate text files containing unique tokens of questions, programs and answers.


```sh
python scripts/preprocess/build_vocabulary.py \
    --clevr-jsonpath data/CLEVR_train_questions.json \
    --output-dirpath data/clevr_vocabulary
```

3. Tokenize programs, questions and answers of CLEVR training, validation (and test) splits using
   this vocabulary mapping. This will create H5 files to be read by [`probnmn.data.readers`][4].

```sh
python scripts/preprocess/preprocess_questions.py \
    --clevr-jsonpath data/CLEVR_train_questions.json \
    --vocab-dirpath data/clevr_vocabulary \
    --output-h5path data/clevr_train_tokens.h5 \
    --split train
```

4. Extract image features using pre-trained ResNet-101 from torchvision model zoo.

```sh
python scripts/preprocess/extract_features.py \
    --image-dir data/images/train \
    --output-h5path data/clevr_train_features.h5 \
    --split train
```


[1]: https://arxiv.org/abs/1902.07864
[2]: https://conda.io/docs/user-guide/install/download.html
[3]: https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
[4]: https://kdexd.github.io/probnmn-clevr/probnmn/data.readers.html
