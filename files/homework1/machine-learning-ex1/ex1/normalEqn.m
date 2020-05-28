function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

% size(X,2) returns the number of colums in X - the second dimension of X
% size(X) returns 47 3 - the number of lines and columns, and
% size(X,1) returns the number of lines in X
theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

[theta] = pinv(X' * X) * X' * y;


% -------------------------------------------------------------


% ============================================================

end
