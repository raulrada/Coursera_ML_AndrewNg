function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % compute m x 1 vector holding predicted values of houses, based on n 
    % independent variables
    % X is an m x (n+1) matrix where X(m,1) = 1 (first column elements are equal 
    % to 1)
    % hypothesis function val i = X(i,1) * theta(1) + ... + X(i,n) * theta(n)
    predictions = X * theta; 
    
    % m x 1 vector holding differences between predicted and actual house prices
    errors = predictions - y;
    
    % delta(j) = 1/m * ((errors(1) * X(1,j) + ... + errors(m) * X(m,j))
    % basically, compute the sum of the products of the error terms for each
    % training example and each X value in column j, in order to get delta(j)
    % transpose errors vector (get row vector) in order to multiply it with each
    % column in X, so as to obtain a row vector with the deltas)
    deltatransposed = 1/m * errors' * X;
    
    % for ease of computation, the delta vector shoud be a column vector
    delta = deltatransposed';
    
    theta = theta - alpha * delta;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
