function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% compute h - the mx1 vector holding the values of the hypothesis function for 
% all training examples;
% X is an m x 2 matrix; theta is a 2 x 1 vector; predictions is an m x 1 vector;
% y is an m x 1 vector
predictions = X * theta;

% J = 1 / (2*m) * squared errors of the predictions + ...
% lambda / (2*m) * sum(squared thetas); theta0 is not regularized!!!
% errors of the predictions equals prediction minus actual value (y).
J = 1 / (2 * m) * sum((predictions - y) .^2) + ...
    lambda / (2 * m) * sum(theta(2:end) .^2);

% the regularization factor for theta is lambda / m * theta(j), for j !=0;
% we do not regularize theta(0), therefore the first element in 
% regularized_theta must be 0!!! theta(1) through theta(n) must be regularized!
grad = 1 / m * X' * (predictions - y) + lambda / m * [0; theta(2:end)];

% =========================================================================

grad = grad(:);

end
