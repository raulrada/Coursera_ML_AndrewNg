function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% need to do element-wize division, and then element-wize power raising!!!
% basically, the function takes a matrix z as parameter, and it returns
% 1 / (1 + e^(-x)) (where x is each element in matrix z)
g = 1 ./ (1 + e.^(-z));



% =============================================================

end
