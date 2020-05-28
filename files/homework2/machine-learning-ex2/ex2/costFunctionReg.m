function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% compute h - the mx1 vector holding the values of the hypothesis function for 
% all training examples
predictions = sigmoid(X * theta);

% don't regularize theta0 (first element, at index 1, in theta vector!!!)
J = -1/m * (y' * log(predictions) + (1 - y)' * log(1 - predictions)) + ...
    lambda / (2*m) * sum(theta(2:end,:) .^ 2);

% the regularization factor for theta is lambda / m * theta(j), for j !=0;
% we do not regularize theta(0), therefore the first element in 
% regularized_theta must be 0!!! theta(1) through theta(n) must be regularized!
regularized_theta = [0; (lambda / m) .* theta(2:end)];
    
grad = 1/m * (X' * (predictions - y)) + regularized_theta;


% =============================================================

end
