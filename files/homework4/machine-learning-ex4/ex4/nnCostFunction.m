function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% The number of elements in the vector which are part of Theta1 is equal to
% the number of rows of Theta1 (that is, hiddel_layer_size - the number of elements
% in the hidden layer) multiplied by the number of columns in Theta1 (that is, 
% the number of elements in the input layer + 1, for the bias unit = +1
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


                 
% Setup some useful variables
m = size(X, 1);

% add first column of ones in X
X = [ones(m, 1) X];

         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% iterate through the training images
for i = 1:m
  % in order to compute the activation nodes in the hidden layer, multiply each
  % line in Theta1 by a column vector obtained by transposing the current line
  % in matrix X, and get the sigmoid of the respective results;
  a2 = sigmoid(Theta1 * X(i, :)');
  
  % add the bias unit (= +1) in the hidden layer, before moving forward
  a2 = [1; a2];
  
  % compute the output layer
  a3 = sigmoid(Theta2 * a2);
  
  % predictions is a column vector, with one element equal to 1 (at the index 
  % equal to the predicted label; for instance, if the maximum probability belongs
  % to a label of 3, the predictions vector (for 10 labels, in our case) is:
  % [0; 0; 1; 0; 0; 0; 0; 0; 0; 0]
  predictions = (a3 == max(a3));
  
  % y(i) is a numberr - the correct label for the current picture. But I need
  % a column vector with all values 0, except for the one at the index corresponding
  % to the correct label for the current image. For instance, if y(i) = 5, then
  % correct_result = [0; 0; 0; 0; 1; 0; 0; 0; 0; 0]
  correct_result = zeros(size(a3, 1), 1);
  correct_result(y(i)) = 1;
  
  % update the cost function value with the cost of the current prediction
  J = J + (-1/m * (correct_result' * log(a3) +... 
      (1 - correct_result)' * log(1 - a3))); 
      
  delta3 = a3 - correct_result;
  Theta2_grad = Theta2_grad + delta3 * a2';
  
  delta2 = Theta2' * delta3 .* (a2 .* (1 - a2)); 
  delta2 = delta2(2:end);
  
  Theta1_grad = Theta1_grad + delta2 * X(i, :);
  
endfor

% add the regularization elements - the sums of the squares of each element in
% Theta1 and Theta2 (excluding the first columns in each Theta matrix - which
% correspond to the bias units!!!!!!!!!!) multiplied by lambda / (2*m)
J = J + lambda / (2*m) * (sum(sum(Theta1(:, 2:end) .^ 2)) + ...
    sum(sum(Theta2(:, 2:end) .^ 2)));

Theta2_reg = (lambda/m) * [zeros(size(Theta2, 1),1) Theta2(:, 2:end)];
Theta2_grad = 1/m * Theta2_grad + Theta2_reg;

Theta1_reg = (lambda/m) * [zeros(size(Theta1, 1),1) Theta1(:, 2:end)];
Theta1_grad = 1/m * Theta1_grad + Theta1_reg;
              
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
