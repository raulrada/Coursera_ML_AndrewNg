function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

% First column of X_poly should be X
X_poly(:, 1) = X;

% Each column in X_poly, from 2 to p, should contain the elements of vector X
% raised to power i
for i = 2:p
  X_poly(:, i) = X(:, 1) .^ i;
endfor


% =========================================================================

end
