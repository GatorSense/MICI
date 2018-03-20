function [fitness] = evalFitness_noisyor(Labels, measure, nPntsBags, oneV, bag_row_ids, diffM, C1, mean, sigma)
% Evaluate fitness for noisy-or model
%
% INPUT
%    Labels         - 1xNumTrainBags double  - Training labels for each bag
%    measure        -  measure to be evaluated after update
%    nPntsBags      - 1xNumTrainBags double    - number of points in each bag
%    oneV           -  - marks where singletons are
%    bag_row_ids    - the row indices of  measure used for each bag
%    diffM          - Precompute differences for each bag
%    C1             - the scaling value in the objective function Gaussian distribution C1 = 1/(sqrt(2*pi) .* Parameters.sigma) / atmean;
%    mean           - mean of Gaussian distribution
%    sigma          - sigma of Gaussian distribution
%
% OUTPUT
%   fitness         - the fitness value using noisy-or objective function (for two-class labels)
%
% Written by: X. Du 03/2018
%


singletonLoc = (nPntsBags == 1);
fitness = 0;

if sum(singletonLoc) %if there are singleton bags

diffM_s = vertcat(diffM{singletonLoc});
bag_row_ids_s = vertcat(bag_row_ids{singletonLoc});
% labels_s = Labels(singletonLoc)';


%Compute CI for singleton bags
ci = sum(diffM_s.*horzcat(measure(bag_row_ids_s),ones(sum(singletonLoc),1)),2);
npdf = C1*exp(-0.5 * ((ci - mean)./sigma).^2);

npdf_label0 = npdf(Labels(singletonLoc)==0);
npdf_label1 = npdf(Labels(singletonLoc)==1);

fitness = fitness + sum(log(1 - npdf_label0 + eps));
fitness = fitness + log(1-exp(sum(log(1 - npdf_label1 + eps))) + eps);

end

diffM_ns = diffM(~singletonLoc);
bag_row_ids_ns = bag_row_ids(~singletonLoc);
labels_ns = Labels(~singletonLoc);
oneV = oneV(~singletonLoc);

%Compute CI for non-singleton bags
for i = 1:length(diffM_ns)
    ci = sum(diffM_ns{i}.*horzcat(measure(bag_row_ids_ns{i}), oneV{i}),2);
    npdf = C1*exp(-0.5 * ((ci - mean)./sigma).^2);

    if(labels_ns(i) ~= 1)
        fitness = fitness + sum(log(1 - npdf + eps));
    else
        fitness = fitness + log(1-exp(sum(log(1 - npdf + eps))) + eps);
    end
end

    % sanity check
    if isinf(fitness) || ~isreal(fitness)
%         keyboard;
        fitness = real(fitness);
        if fitness == Inf
            fitness = -10000000;
        elseif fitness == -Inf
            fitness = -10000000;
        end
    end
    
end