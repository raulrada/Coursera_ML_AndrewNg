function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------

% Use a lambda of 0 for the training set, when computing the value of the 
% training cost function (since lambda was used when training the model and 
% for obtaining theta, we would effectively double the regularization if we
% were to apply a non-zero lambda when computing the cost function based on the
% theta obtained from the training model)
lambda_training = 0;

% Use a lambda of 0 for the training set, when computing the value of the 
% cross validation cost function (same explanation as for lambda_training
lambda_cv = 0;

% Obtain the training and cross validation errors, based on models trained using
% a variable size of the training set, from 1 element up to all elements in the
% training set
for i = 1:m
  
  % Get a training subset containing 1, 2, ..., m elements out of the entire 
  % training set
  Xtraining = X(1:i, :);
  ytraining = y(1:i);
  
  % initialize theta before training the model
  theta = zeros(size(Xtraining, 2), 1);
  
  % Train the model based on the training sub-set selected from the entire 
  % training set
  theta = trainLinearReg(Xtraining, ytraining, lambda);
  
  % With the theta obtained by training the model based on the sub-set of 
  % elements selected from the entire training set, obttain the train and
  % cross validation errors (the values of the cost functions)
  % Be careful to use a lambda of 0 when computing error_train and error_val;
  % we already used lambda when training the model (for obtaining theta), so
  % we should not do the regularization twice, by using a non-zero lambda when
  % computing the cost functions
  error_train(i) = linearRegCostFunction(Xtraining, ytraining, theta,...
                   lambda_training);
  error_val(i) = linearRegCostFunction(Xval, yval, theta,...
                   lambda_cv);   
                   
endfor









% -------------------------------------------------------------

% =========================================================================

end
