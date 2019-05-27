Probabilistic Neural-symbolic Models
====================================

Code for our ICML 2019 paper:

**[Probabilistic Neural-Symbolic Models for Interpretable Visual Question Answering][1]**  
Ramakrishna Vedantam, Karan Desai, Stefan Lee, Marcus Rohrbach, Dhruv Batra, Devi Parikh

Checkout our package documentation at
[kdexd.github.io/probnmn-clevr](https://kdexd.github.io/probnmn-clevr)!

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

Usage Instructions
------------------

1. [How to setup this codebase?][2]
2. [How to train your ProbNMN?][3]
3. [How to evaluate or run inference?][4]


Pre-trained Checkpoint
----------------------

Pre-trained checkpoints and corresponding config files (with all the
hyper-parameters) for all training phases is available with v1.0 release of
this repository. Check out the [Releases](https://github.com/kdexd/probnmn-clevr/releases)!


Acknowledgments
---------------

We thank the developers of:

1. [@davidmascharka/tbd-nets](https://www.github.com/davidmascharka/tbd-nets)
   for providing a very clean implementation of our core Neural Module Network.

2. [@allenai/allennlp](https://www.github.com/allenai/allennlp) for providing
   an awesome framework which indeed takes _masking and padding seriously._ 

3. [@rbgirshick/yacs](https://www.github.com/rbgirshick/yacs) for providing an
   efficient package-wide configuration management.

4. [@pytorch/pytorch](https://www.github.com/pytorch/pytorch), this needs no
   reason.


[1]: https://arxiv.org/abs/1902.07864
[2]: https://kdexd.github.io/probnmn-clevr/probnmn/usage/setup_dependencies.html
[3]: https://kdexd.github.io/probnmn-clevr/probnmn/usage/training.html
[4]: https://kdexd.github.io/probnmn-clevr/probnmn/usage/evaluation_inference.html
