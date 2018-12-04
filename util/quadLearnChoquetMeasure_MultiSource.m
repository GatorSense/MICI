function g = quadLearnChoquetMeasure_MultiSource(H, Y)
%   g = quadLearnChoquetMeasure(H, Y)   for *Any Number (>3)* of sources
%
%   Purpose: Learn the fuzzy measures of a choquet integral for fusing sources
%           of information. Learning the measures is framed as the 
%           following quadratic programming problem:
%
%
%      argmin { 0.5*g'D+gamma'*g }
%         g
%
%       s.t. 0 <= g <=1 and A*g <= b --> A*g - b <= 0  
%
%   Input: H = [n_samp x nSources] Matrix: Support info. [0 1].
%              Each row of H (denoted h) is the confidence from each source
%              of the sample belonging to class 1. 
%          Y = [n_samp x 1] Row Vector: Binary desired output (1 = class 1, 0 = class 0.)
%               
%   Output: g - [2^nSources - 2 x 1] Row vector: Learned fuzzy measures.   
% Written by: X. Du 05/11/2017

%%

% Extract Dimensions of the data.
[nSample, nSources] = size(H);

% Currently only designed for 3 sources...
if(nSources <3)
    error('This code is only meant for the fusion of more then three inputs!');
end
%================== Compute Coefficients ==================================
sec_start_inds = zeros(nSources,1);
nElem_prev = 0;
for j = 1:(nSources-1)
    if j == 1 %singleton        
        sec_start_inds(j) = 0;
        MeasureSection.NumEach(j) = nSources;        % set up MeasureSection (singleton)
        MeasureSection.Each{j} = [1:nSources]';
    else  %non-singleton
        nElem_prev = nElem_prev+MeasureSection.NumEach(j-1);
        sec_start_inds(j) = nElem_prev;
        MeasureSection.NumEach(j) = nchoosek(nSources,j);%compute the cumulative number of measures of each tier. E.g. singletons,2s, 3s,..
        MeasureSection.Each{j} =  nchoosek([1:nSources],j);
    end
end

MeasureSection.NumEach(nSources) = 1;%compute the cumulative number of measures of each tier. E.g. singletons,2s, 3s,..
MeasureSection.Each{nSources} =  [1:nSources];
MeasureSection.NumCumSum = cumsum(MeasureSection.NumEach);

% %Precompute differences and indices for all points
%%%%bag_row_ids --- contains which elements were used in the original data
%%%%for each data point

%================== Compute Z matrix Coefficients ==================================

%       Z = [nSample x 2^nSources - 2] Matrix: Stores the change in
%               values from different sources - h(x_i*) - h(x_i+1*)
%           Each row (z) is a measure of the difference between each source
%           for each sample. gamma(j) in the text. 
Z = zeros(nSample, 2^nSources - 2);

     bag_row_ids = zeros(nSample,nSources-1);
    [v, indx] = sort(H, 2, 'descend');
    %%% vz = horzcat(zeros(size(v,1), 1), v) - horzcat(v, zeros(size(v,1), 1));
   %%% diffM{1} = vz(:, 2:end);
    for j = 1:(nSources-1) %# of sources in the combination (e.g. j=1 for g_1, j=2 for g_12)
            elem = MeasureSection.Each{j};%the number of combinations, e.g., (1,2),(1,3),(2,3)
        for n = 1:size(H,1)          
            if j == 1
                bag_row_ids(n,j) = indx(n,1);
            else  %non-singleton
                temp = sort(indx(n,1:j), 2);
                [~,~,row_id] = ismember_findrow_mex_my(temp,elem);                
                bag_row_ids(n,j) = sec_start_inds(j) + row_id;
            end
            
        gelemIdx = bag_row_ids(n,j); 
        Z(n,gelemIdx) = v(n,j) - v(n,j+1); 
        end
    end



% D = [2^nSources - 2 x 2^n_source - 2] Matrix: 2 * sum over all samples of z'*z, 2 is for scaling purposes.                                               
D = zeros(2^nSources-2 ,2^nSources-2 );
for j = 1:nSample
    D = D + Z(j,:)'*Z(j,:); 
end
D = D*2;

% Compute Gamma over all samples. 
Gamma = zeros(2^nSources-2, 1);
for j = 1:nSample
    Gamma = Gamma + 2*(min(H(j,:))-Y(j))*Z(j,:)'; % Gamma = [2^nSources-2 x 1] column vector: sum(2*(h(x_1*) - y)*z) over all samples.
end





%============ Compute lower bound and upper bound ======================================
% Compute lower bound and upper bound of each element, for compute Contraints later
% %%%% compute lower bound index, correct but not needed! Can also be used
%%%%%% to compute constraints...
% nElem_prev = 0;
% nElem_prev_prev=0;
% lowerindex =[];
% for i = 2:nSources-1 %sample rest
%     nElem = MeasureSection.NumEach(i);%the number of combinations, e.g.,3
%     elem = MeasureSection.Each{i};%the number of combinations, e.g., (1,2),(1,3),(2,3)
%     nElem_prev = nElem_prev+MeasureSection.NumEach(i-1);
%     if i==2
%         nElem_prev_prev = 0;
%     elseif i>2
%         nElem_prev_prev  = nElem_prev_prev + MeasureSection.NumEach(i-2);
%     end
%     for j = 1:nElem
%         lowerindex{nElem_prev+j}  =[];
%         elemSub = nchoosek(elem(j,:), i-1);
%         for k = 1:length(elemSub) %it needs a length(elemSub) because it is taking subset of the element rows
%             tindx = elemSub(k,:);
%             [Locb] = ismember_findRow(tindx,MeasureSection.Each{i-1});
%             lowerindex{nElem_prev+j} = horzcat(lowerindex{nElem_prev+j} , nElem_prev_prev+Locb);
%         end
%     end
% end
upperindex = [];
%%%%compute upper bound index
nElem_nextsum = 2^nSources-1;%total length of measure
for i = nSources-2:-1:1 %sample rest
    nElem = MeasureSection.NumEach(i);%the number of combinations, e.g.,3
    elem = MeasureSection.Each{i};%the number of combinations, e.g., (1,2),(1,3),(2,3)
    %nElem_next = MeasureSection.NumEach(i+1);
    elem_next = MeasureSection.Each{i+1};
    
    nElem_nextsum = nElem_nextsum - MeasureSection.NumEach(i+1); %cumulative sum of how many elements in the next tier so far
    for j = nElem:-1:1
        upperindex{nElem_nextsum-nElem+j-1}  =[];
        elemSub = elem(j,:); 
        [~,~,row_id1] = ismember_findrow_mex_my(elemSub,elem_next);
        upperindex{nElem_nextsum-nElem+j-1} = horzcat(upperindex{nElem_nextsum-nElem+j-1} , nElem_nextsum+row_id1-1);
    end
end


%============ Setup Costraint Matrix ======================================
%   Contraints follow the form:
%   
%           A*g <= b           
%
%   Monotinicity Constraint on g:
%       g{xi} <= g{xi, xj} for any j, following from choquet set constraints.
%
% Compact binary representation of the constraint. 
% A total of nSources*(2^(nSources-1)-1) constraints (ref: J.M.Keller book chapter)

% Set A: 
A = zeros(nSources*(2^(nSources-1)-1), 2^nSources-2);
numConstraints = size(A,1);
numMeasureElems = size(A,2); %number of measure elements with constraints (excluding empty 0 and all 1 sets)

%%%% last nSources rows, g_12..(n-1)<=1 constraints%%%%%%%
count = 0;
for i = (numConstraints-nSources+1) : numConstraints
    count = count+1;
    A(i,numMeasureElems-nSources+count) = 1;
end
%%%% 1: nSources-1 rows, g_12..(n-1)<=1 constraints%%%%%%%
A_row_indx_prev = 0; 
for i = 1:numel(upperindex)
 %BTW: the sum of all elements in "upperindex" is equal to the number of constraints!
 upperindex_numel_temp = numel(upperindex{i});
 A( A_row_indx_prev+1 : A_row_indx_prev+upperindex_numel_temp ,i) = 1;
 for j = 1:upperindex_numel_temp
    A( A_row_indx_prev+j ,upperindex{i}(j)) = -1;
 end
 A_row_indx_prev = A_row_indx_prev+upperindex_numel_temp;
end

% Set b: Refer to comments above. 
b = zeros(nSources*(2^(nSources-1)-1),1);
b([end-nSources+1:end]) = 1; % the last tier constraints <=1

% Use matlab built-in function for solving the quadratic problem.
% options = optimset('Algorithm', 'interior-point-convex');
g = quadprog(D, Gamma, A, b, [], [], zeros(2^nSources-2,1),ones(2^nSources-2,1), []);
% options = optimset('Algorithm', 'interior-point-convex');
% options = optimset('Algorithm', 'trust-region-reflective');
% g = quadprog(D, Gamma, A, b, [], [], [], [], []);
g = [g;1]'; % add the last measure element (always=1)

end
