How to setup this codebase?
===========================

This codebase requires Python 3.6+ or higher. It uses PyTorch v1.0, and has out
of the box support with CUDA 9 and CuDNN 7. The recommended way to set this
codebase up is through Anaconda or Miniconda, although this should work just as
fine with VirtualEnv.

Install Dependencies
--------------------

For these steps to install through Anaconda / Miniconda.

1. Install Anaconda or Miniconda distribution based on Python3+ from their
   `downloads site <https://conda.io/docs/user-guide/install/download.html>`_.


2. Clone the repository first.

    .. code-block:: shell

        git clone https://www.github.com/kdexd/probnmn-clevr


3. Create a conda environment and install all the dependencies.

    .. code-block:: shell

        cd probnmn-clevr
        conda create -n probnmn python=3.6
        conda activate probnmn
        pip install -r requirements.txt


4. Install this codebase as a package in development version.

    .. code-block:: shell

        python setup.py develop


Now you can ``import probnmn`` from anywhere in your filesystem as long as you
have this conda environment activated.


Download and Preprocess Data
----------------------------

1. This codebase assumes all the data to be in ``$PROJECT_ROOT/data`` directory
   by default, although custom paths can be provided through config. Download
   CLEVR v1.0 dataset from `here <https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip>`_
   and symlink it as follows:

    .. code-block:: text

        $PROJECT_ROOT/data
            |—— CLEVR_test_questions.json
            |—— CLEVR_train_questions.json
            |—— CLEVR_val_questions.json
            +—— images
                |—— train
                |   |—— CLEVR_train_000000.png
                |   +—— CLEVR_train_000001.png ...
                |—— val
                |   |—— CLEVR_val_000000.png
                |   +—— CLEVR_val_000001.png ...
                +—— test
                    |—— CLEVR_test_000000.png
                    +—— CLEVR_test_000001.png ...


2. Build a vocabulary out of CLEVR programs, questions and answers, which is
   compatible with AllenNLP, and will be used throughout the training and
   evaluation. This will create a directory with separate text files containing
   unique tokens of questions, programs and answers.

    .. code-block:: shell

        python scripts/preprocess/build_vocabulary.py \
            --clevr-jsonpath data/CLEVR_train_questions.json \
            --output-dirpath data/clevr_vocabulary


3. Tokenize programs, questions and answers of CLEVR training, validation (and
   test) splits using this vocabulary mapping.

    .. code-block:: shell

        python scripts/preprocess/preprocess_questions.py \
            --clevr-jsonpath data/CLEVR_train_questions.json \
            --vocab-dirpath data/clevr_vocabulary \
            --output-h5path data/clevr_train_tokens.h5 \
            --split train


4. Extract image features using pre-trained ResNet-101 from torchvision model
   zoo.

    .. code-block:: shell

        python scripts/preprocess/extract_features.py \
            --image-dir data/images/train \
            --output-h5path data/clevr_train_features.h5 \
            --split train

That's it! Steps 3 and 4 will create necessary H5 files which can be used by
``probnmn.data.readers`` and further ``probnmn.data.datasets``.
