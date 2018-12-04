# MICI:
**Multiple Instance Choquet Integral for Classifier Fusion and Regression**

_Xiaoxiao Du and Alina Zare_

[[`IEEEXplore (MICI Classifier Fusion paper)`](https://ieeexplore.ieee.org/document/7743905)]

[[`IEEEXplore (MICI Classifier Fusion and Regression paper)`](https://ieeexplore.ieee.org/document/8528500)]

[[`arXiv`](https://arxiv.org/abs/1803.04048)] 

[[`BibTeX`](#CitingMICI)]


In this repository, we provide the papers and code for the Multiple Instance Choquet Integral (MICI) Classifier Fusion and/or Regression Algorithms.

## Installation Prerequisites

This code uses MATLAB Statistics and Machine Learning Toolbox,
MATLAB Optimization Toolbox and MATLAB Parallel Computing Toolbox.

## Demo

Run `demo_main.m` in MATLAB.

## Main Functions

The MICI Classifier Fusion and Regression Algorithm runs using the following functions.

1. MICI Classifier Fusion (noisy-or model) Algorithm  

```[measure, initialMeasure,Analysis] = learnCIMeasure_noisyor(TrainBags, TrainLabels, Parameters);```

2. MICI Classifier Fusion (min-max model) Algorithm

 ```[measure, initialMeasure,Analysis] = learnCIMeasure_minmax(TrainBags, TrainLabels, Parameters);```
 
3. MICI Classifier Fusion (generalized-mean model) Algorithm

```[measure, initialMeasure,Analysis] = learnCIMeasure_softmax(TrainBags, TrainLabels, Parameters);```

4. MICI Regression (MICIR) Algorithm

```[measure, initialMeasure,Analysis] = learnCIMeasure_regression(TrainBags, TrainLabels, Parameters);```


## Inputs

#The *TrainBags* input is a 1xNumTrainBags cell. Inside each cell, NumPntsInBag x nSources double -- Training bags data.

#The *TrainLabels* input is a 1xNumTrainBags double vector that takes values of "1" and "0" for two-class classfication problems -- Training labels for each bag. 


## Parameters
The parameters can be set in the following function:

```[Parameters] = learnCIMeasureParams();```

The parameters is a MATLAB structure with the following fields:
1. nPop: size of population
2. sigma: sigma of Gaussians in fitness function
3. maxIterations: maximum number of iterations
4. eta: percentage of time to make small-scale mutation
5. sampleVar: variance around sample mean
6. mean: mean of CI in fitness function. This value is always set to 1 (or very close to 1) if the positive label is "1".
7. analysis: if ="1", save all intermediate results
8. p: the power coefficient for the generalized-mean function. Empirically, setting p(1) to a large postive number and p(2) to a large negative number works well. 

*Parameters can be modified by users in [Parameters] = learnCIMeasureParams() function.*


## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2018 X. Du and A. Zare. All rights reserved.

## <a name="CitingMICI"></a>Citing MICI

If you use the MICI clasifier fusion and regression algorithms, please cite the following references using the following BibTeX entries.
```
@ARTICLE{du2018multiple,
author={X. Du and A. Zare},
journal={IEEE Transactions on Geoscience and Remote Sensing},
title={Multiple Instance Choquet Integral Classifier Fusion and Regression for Remote Sensing Applications},
year={2018},
volume={},
number={},
pages={1-13},
doi={10.1109/TGRS.2018.2876687}
}
```
```
@INPROCEEDINGS{du2016multiple,
author={X. Du and A. Zare and J. M. Keller and D. T. Anderson},
booktitle={IEEE Congress on Evolutionary Computation (CEC)},
title={Multiple Instance Choquet integral for classifier fusion},
year={2016},
volume={},
number={},
pages={1054-1061},
doi={10.1109/CEC.2016.7743905},
month={July}
}
```

## <a name="Related Work"></a>Related Work

Also check out our Multiple Instance Multi-Resolution Fusion (MIMRF) algorithm for multi-resolution fusion!


[[`arXiv`](https://arxiv.org/abs/1805.00930)] 

[[`GitHub Code Repository`](https://github.com/GatorSense/MIMRF)] 
