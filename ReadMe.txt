MICI (Multiple Instance Choquet Integral)  for Classifier Fusion and Regression       ----- ReadMe File

This folder Includes papers and demo MATLAB code for:

(1) MICI Classifier Fusion (noisy-or model) Algorithm  
(2) MICI Classifier Fusion (min-max model) Algorithm
(3) MICI Classifier Fusion (generalized-mean model) Algorithm
(4) MICI Regression (MICIR) Algorithm

***************************************************************
***NOTE: If the MICI Classifier Fusion and/or Regression Algorithms are used in any publication or presentation, the following reference must be cited:

 [1] X. Du, A. Zare, J. Keller, D. Anderson, "Multiple Instance Choquet Integral for Classifier Fusion," 2016 IEEE Congress on Evolutionary Computation (CEC), Vancouver, BC, 2016, pp. 1054-1061.
 [2] X. Du and A. Zare, "Multiple Instance Choquet Integral Classifier Fusion and Regression for Remote Sensing Applications," Under Review. Available: https://arxiv.org/abs/1803.04048

***************************************************************
***
The MICI Classifier Fusion and Regression Algorithm runs using the following function:

[Parameters] = learnCIMeasureParams();
[measure, initialMeasure,Analysis] = learnCIMeasure_noisyor(TrainBags, TrainLabels, Parameters);
OR    [measure, initialMeasure,Analysis] = learnCIMeasure_minmax(TrainBags, TrainLabels, Parameters);
OR    [measure, initialMeasure,Analysis] = learnCIMeasure_softmax(TrainBags, TrainLabels, Parameters);
OR    [measure, initialMeasure,Analysis] = learnCIMeasure_regression(TrainBags, TrainLabels, Parameters);

The TrainBags input is a 1xNumTrainBags cell. Inside each cell, NumPntsInBag x nSources double. Training bags data.
The TrainLabels input is a 1xNumTrainBags double vector that only take values of "1" and "0" (two-class classfication problems). Training labels for each bag.

The parameters input is a struct with the following fields: 
  Parameters - struct - The struct contains the following fields:
                 These parameters are user defined (can be modified)
                    1. nPop: Size of population
                    2. sigma: Sigma of Gaussians in fitness function
                    3. nIterations: Number of iterations
                    4. eta:Percentage of time to make small-scale mutation
                    5. sampleVar: Variance around sample mean
                    6. mean: mean of ci in fitness function. Is always set to 1 if the positive label is "1".
                    7. analysis: if ="1", record all intermediate results
                 *Parameters can be modified in [Parameters] = learnCIMeasureParams() function.

********************************************************************* 
****The root directory contains the following files:

demo_main.m                     - Run this. This is the main demo file for all classifier fusion and regression models. 
demo_data_cl.mat              - Example 3-source 2-class classification data (provided for demo purposes)
	             In this demo data set: Bags: 1x50 cell - 50 bags, each bag has 20 data points with 3 sources
                                    gtrue: true measure
                                    Labels: 1x50 - bag-level labels
                                    X: for plotting (2-D visualization) purposes
learnCIMeasureParams.m              - User-set Parameters

****The subfolders contains the following files:
./papers     - contains all publications associated with the MICI algorithms
./util            - All functions, as follows:
ChoquetIntegral_g_MultiSources.m   - Compute the Choquet integral output given a single data point "hx" and measure "g"
computeci.c                        - Compute the Choquet integral output given a data point "hx" and measure "g" in c code. *Need to run "mex computeci.c"
ismember_findrow_mex.c	- Find row index if vector A is part of a row in Vector B in c code. **Need to run in MATLAB "mex ismember_findrow_mex.c"
ismember_findrow_mex_my.m     - Output row index if vector A is part of a row in Vector B (uses above c code).
share.h    - global variable header to be used in computeci.c
learnCIMeasure_noisyor.m            - MICI Two-Class Classifier Fusion algorithm function with noisy-or objective function.
learnCIMeasure_noisyor_CountME1.m            - MICI Two-Class Classifier Fusion algorithm function with noisy-or objective function using ME optimization.
evalFitness_noisyor.m            - Fitness function for MICI Two-Class Classifier Fusion algorithm function with noisy-or objective function.
learnCIMeasure_minmax.m            - MICI Two-Class Classifier Fusion algorithm function with min-max objective function.
evalFitness_minmax.m            - Fitness function for MICI Two-Class Classifier Fusion algorithm function with min-max objective function.
learnCIMeasure_softmax.m            - MICI Two-Class Classifier Fusion algorithm function with generalized-mean objective function.
evalFitness_softmax.m            - Fitness function for MICI Two-Class Classifier Fusion algorithm function with generalized-mean objective function.
learnCIMeasure_regression.m            - MICI Regression algorithm function.
evalFitness_reg.m            - Fitness function for MICIR with regression objective function.
invcdf_TruncatedGaussian.m       - compute inverse cdf for Truncated Gaussian.
sampleMeasure.m                  - either flip a coin and randomly sample a brand new measure from uniform between sampleMeasure_Above and sampleMeasure_Bottom; OR only sample and update one element in the measure.
sampleMeasure_Above.m            - sampling a new measure from "top-down".
 sampleMeasure_Bottom.m           - sampling a new measure from "bottom-up".
sampleMultinomial_mat.m      - sample from a multinomial distribution.

quadLearnChoquetMeasure_MultiSource.m   -- code for the CI-QP method (learn CI measures using quadratic programming) for multiple (>=3) sources
quadLearnChoquetMeasure_3Source.m     --code for CI-QP method, hard-coded for 3 sources
quadLearnChoquetMeasure_4Source.m     --code for CI-QP method, hard-coded for 4 sources
quadLearnChoquetMeasure_5Source.m     --code for CI-QP method, hard-coded for 5 sources
********************************************************************* 
***********************************************************************
Authors: Xiaoxiao Du, Alina Zare
Department of Electrical and Computer Engineering, University of Missouri
Department of Electrical and Computer Engineering, University of Florida
 Email Address: xdy74@mail.missouri.edu; azare@ece.ufl.edu
 Latest Revision: March 2018

This code uses MATLAB Statistics and Machine Learning Toolbox, 
MATLAB Optimization Toolbox and MATLAB Parallel Computing Toolbox. 

% This product is Copyright (c) 2018 X. Du and A. Zare
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions
% are met:
%
% 1. Redistributions of source code must retain the above copyright
% notice, this list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright
% notice, this list of conditions and the following disclaimer in the
% documentation and/or other materials provided with the distribution.
% 3. Neither the name of the University nor the names of its contributors
% may be used to endorse or promote products derived from this software
% without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF MISSOURI AND
% CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,
% INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OR CONTRIBUTORS
% BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
% LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES,
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
% HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
% OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
