function g = quadLearnChoquetMeasure_3Source(H, Y)
%   g = quadLearnChoquetMeasure(H, Y) 
%   This code only works with 3 sources
%   
%   Purpose: Learn the fuzzy measures of a choquet integral for fusing sources
%           of information. Learning the measures is framed as the 
%           following quadratic programming problem:
%
%
%      argmin { 0.5*g'Dg+gamma'*g }
%         g
%
%       s.t. 0 <= g <=1 and A*g <= b --> A*g - b <= 0  
%
%   Input: H = [n_samp x n_sources] Matrix: Support info. [0 1].
%              Each row of H (denoted h) is the confidence from each source
%              of the sample belonging to class 1. 
%          Y = [n_samp x 1] Row Vector: Binary desired output (1 = class 1, 0 = class 0.)
%               
%   Output: g - [2^n_sources - 2 x 1] Row vector: Learned fuzzy measures.   
%
%%
% Extract Dimensions of the data.
[n_samp, n_sources] = size(H);

% Currently only designed for 3 sources...
if(n_sources ~=3)
    error('This code is only meant for the fusion of three inputs');
end

%================== Compute Coefficients ==================================

%       Z = [n_samp x 2^n_sources - 2] Matrix: Stores the change in
%               values from different sources - h(x_i*) - h(x_i+1*)
%           Each row (z) is a measure of the difference between each source
%           for each sample. gamma(j) in the text. 
Z = zeros(n_samp, 6);
for j = 1:n_samp
    hx = H(j,:);                                          % Pull a single sample's support info hx = [h{x1} h{x2} h{x3}] 
    [h_xsorted, sort_inds] = sort(hx, 'descend');         % Sort h(x_i) s.t. h(x_1*) > h(x_2*> ...
    Z(j,sort_inds(1)) = h_xsorted(1) - h_xsorted(2);      % Store h(x1*) - h(x2*) 
    g2 = GElement2(sort_inds);                            % Figure out which set combination corresponds to h(x2*)
    Z(j,g2) = h_xsorted(2) - h_xsorted(3);                % Store h(x2*) - h(x3*)
end

% D = [2^n_sources - 2 x 2^n_source - 2] Matrix: 2 * sum over all samples of z'*z, 2 is for scaling purposes.                                               
D = zeros(6,6);
for j = 1:n_samp
    D = D + Z(j,:)'*Z(j,:); 
end
D = D*2;

% Compute Gamma over all samples. 
G = zeros(6, 1);
for j = 1:n_samp
    G = G + 2*(min(H(j,:))-Y(j))*Z(j,:)'; % Gamma = [2^n_sources-2 x 1] column vector: sum(2*(h(x_1*) - y)*z) over all samples.
end

%============ Setup Costraint Matrix ======================================
%   Contraints follow the form:
%   
%           A*g <= b           
%
%   Monotinicity Constraint on g:
%       g{xi} <= g{xi, xj} for any j, following from choquet set constraints.
%
% Compact binary representation of the following constraint. 
% Assume: g = [g{x1} g{x2] g{x3] g{x1,x2} g{x1,x3} g{x2,x3}]
%
% g{x1} - g{x1,x2} <= 0   --> g{x1} <= g{x1,x2}
% g{x1} - g{x1,x3} <= 0   --> g{x1} <= g{x1,x3}
% g{x2} - g{x1,x2} <= 0   --> g{x2} <= g{x1,x2}
% g{x2} - g{x2,x3} <= 0   --> g{x2} <= g{x2,x3}
% g{x3} - g{x1,x3} <= 0   --> g{x3} <= g{x1,x3}
% g{x3} - g{x2,x3} <= 0   --> g{x3} <= g{x2,x3}
%   0   + g{x1,x2} <= 1   --> g{x1,x2} <= 1
%   0   + g{x1,x3} <= 1   --> g{x1,x3} <= 1
%   0   + g{x2,x3} <= 1   --> g{x2,x3} <= 1 

% Set A: 
%      Singletons     |       Combinations
%  g{x1}+g{x2]+g{x3]+g{x1,x2}+g{x1,x3}+g{x2,x3} 
A = [1      0     0      -1        0       0;   % g{x1} - g{x1,x2} 
     1      0     0       0       -1       0;   % g{x1} - g{x1,x3} 
     0      1     0      -1        0       0;   % g{x2} - g{x1,x2}
     0      1     0       0        0      -1;   % g{x2} - g{x2,x3} 
     0      0     1       0       -1       0;   % g{x3} - g{x1,x3}
     0      0     1       0        0      -1;   % g{x3} - g{x2,x3}
     0      0     0       1        0       0;   %   0   + g{x1,x2}
     0      0     0       0        1       0;   %   0   + g{x1,x3}
     0      0     0       0        0       1];  %   0   + g{x2,x3} 

% Set b: Refer to comments above. 
b = [0 0 0 0 0 0 1 1 1]';

% Use matlab built-in function for solving the quadratic problem.
%options = optimset('Algorithm', 'active-set');
 g = quadprog(D, G, A, b, [], [], zeros(2^n_sources-2,1),ones(2^n_sources-2,1), []); %add upper and lower bounds of [0,1] - X. Du 01/13/2016

end

% Determine g{x2*} which density to use as the second weight. Can be thought of as
%   picking the density which coresponds to the combination of sets which
%   gives the most worth. 
function element = GElement2(sort_inds)
    if(sort_inds(1) == 1)
        if(sort_inds(2) == 2)
            element = 4;    % Use g{x1, x2}
        else
            element = 5;    % Use g{x1, x3}
        end
    elseif(sort_inds(1) == 2)
        if(sort_inds(2) == 1)
            element = 4;     % Use g{x1, x2}
        else
            element = 6;     % Use g{x2, x3}
        end
    elseif(sort_inds(1) == 3)
        if(sort_inds(2) == 1)
            element = 5;    % Use g{x1, x3}
        else
            element = 6;    % Use g{x2, x3}
        end
    end
end
