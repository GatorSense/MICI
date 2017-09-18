%% Main demo function for MICI two-class classifier fusion
% Uses function learnCIMeasure_noisyor() for training process
%
%
% REFERENCE :
% X. Du, A. Zare, J. Keller, D. Anderson 
% “Multiple Instance Choquet Integral for Classifier Fusion,”  
% Submitted to 2016 IEEE World Congress on Computational Intelligence (WCCI).
%
% This product is Copyright (c) 2016 University of Missouri.
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions
% are met:
%
%   1. Redistributions of source code must retain the above copyright
%      notice, this list of conditions and the following disclaimer.
%   2. Redistributions in binary form must reproduce the above copyright
%      notice, this list of conditions and the following disclaimer in the
%      documentation and/or other materials provided with the distribution.
%   3. Neither the name of the University nor the names of its contributors
%      may be used to endorse or promote products derived from this software
%      without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF MISSOURI AND
% CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,
% INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED.  IN NO EVENT SHALL THE UNIVERSITY OR CONTRIBUTORS
% BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
% LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES,
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
% HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
% OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%

%% Demo: load 3-source data
% 
% prompt = 'Running this code will clear your workspace.Would you like to continue?[Y/N]';
% str = input(prompt,'s');
% if (str=='N') || (str=='n')
%     ;
% elseif (str=='Y') || (str=='y')
%     
% clear;close all;clc
load('demo_data_cl.mat');

%%%%%%% If no mex file existed in the folder, run the following two lines to compile mex file %%%%%%%
% mex computeci.c
% mex ismember_findrow_mex.c

%%  Training Stage Given TrainBags and TrainLabels
[Parameters] = learnCIMeasureParams();
[measure, initialMeasure,Analysis] = learnCIMeasure_noisyor(Bags, Labels, Parameters);
save('MICI_noisyor_demo_results.mat','measure','Parameters','initialMeasure','Analysis');


%%  Testing Stage Given the above learned measures and the same data
% load('MICI_noisyor_demo_results.mat','measure') % if the learned measure is pre-saved 
Ytrue = computeci(Bags,gtrue);  %true label
Yestimate = computeci(Bags,measure);  % learned measure by MICI
%% Plot true and estimated labels
figure;
set(gcf, 'Position', [0 0 1920 1010]);%get(0,'Screensize'));
subplot(1,2,1);scatter(X(:,1),X(:,2),[],Ytrue);title('True Labels');
subplot(1,2,2);scatter(X(:,1),X(:,2),[],Yestimate);title('MICI Fusion result: Estimated Labels')

% end



