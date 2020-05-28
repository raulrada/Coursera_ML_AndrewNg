function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% m x 1 vector holding predicted profits based on population in each city
predictions = X * theta; 

% m x 1 vector holding the difference between predicted profit and actual profit
errors = predictions - y;

% vector holding the squared errors of prediction for each city
squarederrors = errors .^ 2;

% compute cost function J
J = 1 / (2*m) * sum(squarederrors);

% =========================================================================

end
