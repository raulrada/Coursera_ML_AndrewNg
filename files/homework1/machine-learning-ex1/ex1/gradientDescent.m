function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % m x 1 vector holding profit predictions for each city based on city population
    predictions = X * theta;
    
    % m x 1 vector holding the differences between predicted and actual profits 
    %in each city
    errors = predictions - y;
    
    % get a 1 x 2 vector holding the terms
    % 1/m * sum((predicted value i - actual value i) * xi)
    deltatransposed = 1/m * errors' * X;
    
    % get delta as a column vector
    delta = deltatransposed';
    
    theta = theta - alpha * delta;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
