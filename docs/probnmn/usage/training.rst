How to train your ProbNMN?
==========================

Training a ProbNMN is done in three phases (plus an extra preprocessing-ish
phase). This codebase supports training based on our proposed objective for
ProbNMN, as well as baseline objective according to
`Johnson et al. (CVPR 2017) <https://arxiv.org/abs/1705.03633>`_.

Training is governed by a YAML config file, which has a ``PHASE`` field, with
the name of the training phase. Phase names are corresponding trainers are:

1. ``program_prior``:
   :class:`~probnmn.trainers.program_prior_trainer.ProgramPriorTrainer`
2. ``question_coding``:
   :class:`~probnmn.trainers.question_coding_trainer.QuestionCodingTrainer`
3. ``module_training``:
   :class:`~probnmn.trainers.module_training_trainer.ModuleTrainingTrainer`
4. ``joint_training``:
   :class:`~probnmn.trainers.joint_training_trainer.JointTrainingTrainer`


.. note::

    1. Execute all the commands from ``$PROJECT_ROOT`` to use the config files
       used to reproduce results in the paper. Configuration is managed through
       YAML files, with a central package-wide configuration management system.
       Read more at :class:`~probnmn.config.Config`.

    2. All the training phases will by default serialize checkpoints every few
       iterations, and serialize tensorboard logs every iteration in the same
       directory provided through ``--serialization-dir``. Use ``tensorboard
       --logdir $SERIALIZATION_DIR`` to view training curves, validation
       metrics etc. directly on tensorboard.

    3. The subset of question-program paired training data is selected randomly, hence
       the quality of supervision dataset is governed by random seed. Sometimes it may
       not be the best, and training might be slow. We recommend running ``question_coding``
       and ``joint_training`` for at least 5-7 different random seeds, use
       ``--config-override RANDOM_SEED $NUM``. Having same random seed will ensure the
       selection of same paired data across different run and different machines.

    4. If time / resources are limited, we recommend random seed 700 for decent results.


``PHASE: program_prior``
------------------------

Train a :class:`~probnmn.models.program_prior.ProgramPrior` using all the
programs from CLEVR v1.0 training split. Alternatively, this can be trained
using programs simulated using syntax.

.. code-block::

    python scripts/train.py \
        --config-yml configs/program_prior.yml \
        --phase program_prior \
        --gpu-ids 0 \
        --serialization-dir checkpoints/program_prior

This step does not apply for ``baseline`` objective.


``PHASE: question_coding``
--------------------------

Learn a latent "code" for questions, given some program-question pairs, and a
large amount of questions without paired programs. Choose appropriate config
file according to training objective (``baseline`` or ``ours``).

.. code-block::

    python scripts/train.py \
        --config-yml configs/question_coding_ours.yml \
        --phase question_coding \
        --config-override CHECKPOINTS.PROGRAM_PRIOR checkpoints/program_prior/program_prior_best.pth \
        --gpu-ids 0 \
        --serialization-dir checkpoints/question_coding_ours

Parameters of :class:`~probnmn.models.program_prior.ProgramPrior` are frozen.


``PHASE: module_training``
--------------------------

Train a neural module network with (image, question, answer) tuples, where the
:class:`~probnmn.models.program_generator.ProgramGenerator` trained in
``question_coding`` phase (kept frozen) infers programs from questions.

.. code-block::

    python scripts/train.py \
        --config-yml configs/module_training.yml \
        --phase module_training \
        --config-override CHECKPOINTS.QUESTION_CODING checkpoints/question_coding_ours/question_coding_best.pth \
        --gpu-ids 0 1 2 3 \
        --serialization-dir checkpoints/question_coding_ours

Multi-GPU execution is supported here. This phase is the same for both training
objectives.


``PHASE: joint_training``
-------------------------

.. code-block::

    python scripts/train.py \
        --config-yml configs/joint_training_ours.yml \
        --phase joint_training \
        --config-override CHECKPOINTS.PROGRAM_PRIOR checkpoints/program_prior/program_prior_best.pth \
                          CHECKPOINTS.QUESTION_CODING checkpoints/question_coding_ours/question_coding_best.pth \
                          CHECKPOINTS.MODULE_TRAINING checkpoints/module_training/module_training_best.pth \
        --gpu-ids 0 1 2 3 \
        --serialization-dir checkpoints/joint_training_ours
