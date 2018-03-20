function g = quadLearnChoquetMeasure_4Source(H, Y)
%   g = quadLearnChoquetMeasure(H, Y)   for 4 sources
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
% Written by: X. Du 01/13/2016

%%
% Extract Dimensions of the data.
[nSample, nSources] = size(H);

% Currently only designed for 3 sources...
if(nSources ~=4)
    error('This code is only meant for the fusion of four inputs');
end
%================== Compute Coefficients ==================================

%       Z = [nSample x 2^nSources - 2] Matrix: Stores the change in
%               values from different sources - h(x_i*) - h(x_i+1*)
%           Each row (z) is a measure of the difference between each source
%           for each sample. gamma(j) in the text. 
Z = zeros(nSample, 2^nSources - 2);
for j = 1:nSample
    hx = H(j,:);                                          % Pull a single sample's support info hx = [h{x1} h{x2} h{x3}] 
    [h_xsorted, sort_inds] = sort(hx, 'descend');         % Sort h(x_i) s.t. h(x_1*) > h(x_2*> ...
    Z(j,sort_inds(1)) = h_xsorted(1) - h_xsorted(2);      % Store h(x1*) - h(x2*) 
    g2 = GElement2(sort_inds);                            % Figure out which set combination corresponds to h(x2*)
    Z(j,g2) = h_xsorted(2) - h_xsorted(3);                % Store h(x2*) - h(x3*)
    g3 = GElement3(sort_inds);
    Z(j,g3) = h_xsorted(3) - h_xsorted(4);  
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
%%%%# 1  2  3  4  5  6  7  8  9  10  11  12  13  14 ...
A = [ 1  0  0  0 -1  0  0  0  0   0   0   0   0   0;   %g_1<=g_12
      1  0  0  0  0 -1  0  0  0   0   0   0   0   0 ;   %g_1<=g_13
      1  0  0  0  0  0 -1  0  0   0   0   0   0   0 ;   %g_1<=g_14
      0  1  0  0 -1  0  0  0  0   0   0   0   0   0 ;   %g_2<=g_12   
      0  1  0  0  0  0  0 -1  0   0   0   0   0   0 ;   %g_2<=g_23  
      0  1  0  0  0  0  0  0 -1   0   0   0   0   0 ;   %g_2<=g_24   
      0  0  1  0  0 -1  0  0  0   0   0   0   0   0 ;   %g_3<=g_13   
      0  0  1  0  0  0  0 -1  0   0   0   0   0   0 ;   %g_3<=g_23  
      0  0  1  0  0  0  0  0  0  -1   0   0   0   0 ;   %g_3<=g_34   
      0  0  0  1  0  0 -1  0  0   0   0   0   0   0 ;   %g_4<=g_14   
      0  0  0  1  0  0  0  0 -1   0   0   0   0   0 ;   %g_4<=g_24  
      0  0  0  1  0  0  0  0  0  -1   0   0   0   0 ;   %g_4<=g_34   
      0  0  0  0  1  0  0  0  0   0  -1   0   0   0 ;   %g_12<=g_123
      0  0  0  0  1  0  0  0  0   0   0  -1   0   0 ;   %g_12<=g_124
      0  0  0  0  0  1  0  0  0   0  -1   0   0   0 ;   %g_13<=g_123
      0  0  0  0  0  1  0  0  0   0   0   0  -1   0 ;   %g_13<=g_134  
      0  0  0  0  0  0  1  0  0   0   0  -1   0   0 ;   %g_14<=g_124  
      0  0  0  0  0  0  1  0  0   0   0   0  -1   0 ;   %g_14<=g_134   
      0  0  0  0  0  0  0  1  0   0  -1   0   0   0 ;   %g_23<=g_123   
      0  0  0  0  0  0  0  1  0   0   0   0   0  -1 ;   %g_23<=g_234  
      0  0  0  0  0  0  0  0  1   0   0  -1   0   0 ;   %g_24<=g_124   
      0  0  0  0  0  0  0  0  1   0   0   0   0  -1 ;   %g_24<=g_234   
      0  0  0  0  0  0  0  0  0   1   0   0  -1   0 ;   %g_34<=g_134  
      0  0  0  0  0  0  0  0  0   1   0   0   0  -1 ;   %g_34<=g_234
      0  0  0  0  0  0  0  0  0   0   1   0   0   0 ;   %g_123<=1   
      0  0  0  0  0  0  0  0  0   0   0   1   0   0 ;   %g_124<=1   
      0  0  0  0  0  0  0  0  0   0   0   0   1   0 ;   %g_134<=1  
      0  0  0  0  0  0  0  0  0   0   0   0   0   1 ;   %g_234<=1  
    ];
% Set b: Refer to comments above. 
b = zeros(nSources*(2^(nSources-1)-1),1);
b([end-nSources+1:end]) = 1; % the last tier constraints <=1

% Use matlab built-in function for solving the quadratic problem.
 options = optimset('Algorithm', 'active-set');
g = quadprog(D, Gamma, A, b, [], [], zeros(2^nSources-2,1),ones(2^nSources-2,1), [], options);
% options = optimset('Algorithm', 'interior-point-convex');
% options = optimset('Algorithm', 'trust-region-reflective');
% g = quadprog(D, Gamma, A, b, [], [], [], [], []);

end

% Determine g{x2*} which density to use as the second weight. Can be thought of as
%   picking the density which coresponds to the combination of sets which
%   gives the most worth. 
function element = GElement2(sort_inds)
    if(sort_inds(1) == 1)
        if(sort_inds(2) == 2)
            element = 5;    % Use g{x1, x2}
        elseif(sort_inds(2) == 3)
            element = 6;    % Use g{x1, x3}
        elseif(sort_inds(2) == 4)
            element = 7;    % Use g{x1, x4}
        end
    elseif(sort_inds(1) == 2)
        if(sort_inds(2) == 1)
            element = 5;     % Use g{x1, x2}
        elseif(sort_inds(2) == 3)
            element = 8;     % Use g{x2, x3}
        elseif(sort_inds(2) == 4)
            element = 9;     % Use g{x2, x4}
        end
    elseif(sort_inds(1) == 3)
        if(sort_inds(2) == 1)
            element = 6;    % Use g{x1, x3}
        elseif(sort_inds(2) == 2)
            element = 8;     % Use g{x2, x3}
        elseif(sort_inds(2) == 4)
            element = 10;     % Use g{x3, x4}
        end
    elseif(sort_inds(1) == 4)
        if(sort_inds(2) == 1)
            element = 7;    % Use g{x1, x4}
        elseif(sort_inds(2) == 2)
            element = 9;     % Use g{x2, x4}
        elseif(sort_inds(2) == 3)
            element = 10;     % Use g{x3, x4}
        end
    end
end

% Determine g{x2*} which density to use as the second weight. Can be thought of as
%   picking the density which coresponds to the combination of sets which
%   gives the most worth. 
function element = GElement3(sort_inds)
    if ismember(sort_inds([1:3]),perms([1 2 3]),'rows') == 1
            element = 11;    % Use g{x1, x2, x3}
    end
    
    if ismember(sort_inds([1:3]),perms([1 2 4]),'rows') == 1
            element = 12;    % Use g{x1, x2, x4}
    end
    
    if ismember(sort_inds([1:3]),perms([1 3 4]),'rows') == 1
            element = 13;    % Use g{x1, x3, x4}
    end
    
    if ismember(sort_inds([1:3]),perms([2 3 4]),'rows') == 1
            element = 14;    % Use g{x2, x3, x4}
    end
end
