This is research product by Shunsuke Nakamura as an intern at Preferred Networks.
Training the good model by biased or noisy dataset is difficult.
The goal of this research is developing the robust estimation algorithm for such not clean dataset.

There are 3 method.
- 1. Standart recognizer.
- 2. Learning to Reweight Examples for Robust Deep Learning(LRE)   https://arxiv.org/abs/1803.09050
- 3. Proposed by Shunsuke Nakamura.

## Dependency
chainer
chainer_chemistry


## Installation

```
$ pip install chainer
$ pip install chainer_chemistry
```

## Usage
python train_imbalanced_mnist.py

## option

### --eval-mode
Evaluation mode,
- 0: only binary_accuracy is calculated.
- 1: binary_accuracy and ROC-AUC score is calculated

### --updater-type
- standard: standard updater
- LRE: Learning to Reweight Examples for Robust Deep Learning
- proposed: proposed method
