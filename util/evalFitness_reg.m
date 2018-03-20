
function [fitness] = evalFitness_reg(Labels, measure, nPntsBags, oneV, bag_row_ids, diffM)
% Evaluate the fitness a measure, using min(sum(min((ci-d)^2))) for regression.
%
% INPUT
%    Labels         - 1xNumTrainBags double  - Training labels for each bag
%    measure        -  measure to be evaluated after update
%    nPntsBags      - 1xNumTrainBags double    - number of points in each bag
%    oneV           -  - marks where singletons are
%    bag_row_ids    - the row indices of  measure used for each bag
%    diffM          - Precompute differences for each bag
%
% OUTPUT
%   fitness         - the fitness value using min(sum(min((ci-d)^2))) for regression.
%
% Written by: X. Du 03/2018
%

singletonLoc = (nPntsBags == 1);
fitness = 0;

if sum(singletonLoc) %if there are singleton bags

diffM_s = vertcat(diffM{singletonLoc});
bag_row_ids_s = vertcat(bag_row_ids{singletonLoc});
labels_s = Labels(singletonLoc)';


%Compute CI for singleton bags
ci = sum(diffM_s.*horzcat(measure(bag_row_ids_s),ones(sum(singletonLoc),1)),2);
fitness_s = sum((ci - labels_s).^2);
fitness = fitness - fitness_s;   

end

diffM_ns = diffM(~singletonLoc);
bag_row_ids_ns = bag_row_ids(~singletonLoc);
labels_ns = Labels(~singletonLoc);
oneV = oneV(~singletonLoc);

%Compute CI for non-singleton bags
for i = 1:length(diffM_ns)
    ci = sum(diffM_ns{i}.*horzcat(measure(bag_row_ids_ns{i}), oneV{i}),2);
    fitness = fitness - min((ci-labels_ns(i)).^2);
end

   % sanity check
    if isinf(fitness) || ~isreal(fitness)
        fitness = real(fitness);
        if fitness == Inf
            fitness = -10000000;
        elseif fitness == -Inf
            fitness = -10000000;
        end
    end
    

end

