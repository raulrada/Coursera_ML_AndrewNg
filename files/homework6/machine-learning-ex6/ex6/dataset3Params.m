function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% try out different combinations of values for C and sigma
possible_cs = [0.01; 0.03; 0.1; 0.3; 1.0; 3.0; 10.0; 30.0];
possible_sigma = [0.01; 0.03; 0.1; 0.3; 1.0; 3.0; 10.0; 30.0];

% matrix with length(possible_cs) * length(possible_sigma) lines and 3 columns
% each row is a vector of form [possible_c possible_sigma associated_error]
combinations = [];

% iterate over all candidate values for C and sigma and try out different combinations
for i = 1:length(possible_cs)
  for j = 1:length(possible_sigma)
    % train the model for a given value of C and of sigma, out of the candidate
    % values
    model= svmTrain(X, y, possible_cs(i), @(x1, x2) gaussianKernel(x1, x2, possible_sigma(j)));
    
    % get the predicted values, based on the trained model, applied to the
    % cross-validation examples
    predictions = svmPredict(model, Xval);
    
    % add a new line to combinations matrix, holding the current values of c and
    % of sigma, together with the associated error of the model trained with
    % the current value of c and sigma; error computed based on cross-validation
    % examples !!!
    combinations  = [combinations; possible_cs(i) possible_sigma(j) ...
    mean(double(predictions ~= yval))];
  endfor
endfor

disp('combinations:'), disp(combinations);
% =========================================================================

% get the index of the line in which the minimum error value in column 3 of 
% combinations matrix is found
[~, min_index] = min(combinations(:, 3));

disp('index of min error:'), disp(min_index);

% having the index of the line in matrix combinations which contains the lowest
% error determined on the cross-validation examples, get the C and sigma values\
% from this line in combinations; these are the best values of C and sigma for
% training our model, based on the checking on the cross-validation examples
C = combinations(min_index, 1);
sigma = combinations(min_index, 2);

disp('C for min error:'), disp(C);
disp('sigma for min error:'), disp(sigma);

end
