How to evaluate or do inference?
================================

We provide scripts which support evaluation of any phase, and inference after
joint training phase (on test split).

Evaluation of a particular phase, given its checkpoint can be done as::

    python scripts/evaluate.py \
        --config-yml configs/<name_of_config.yml> \
        --phase <phase_name> \
        --checkpoint-path /path/to/phase_checkpoint.pth \
        --gpu-ids 0


Inference for ``joint_training`` phase, given its checkpoint can be done as::

    python scripts/inference.py \
        --config-yml configs/<name_of_joint_training_config.yml> \
        --checkpoint-path /path/to/joint_training_checkpoint.pth \
        --gpu-ids 0
