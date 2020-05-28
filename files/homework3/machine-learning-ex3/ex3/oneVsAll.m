function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
% m = 5000 (number of training examples);
% n = 400 - number of features (in this case, pixels in each of the training images
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
% PRACTIC, VREAU O MATRICE all_theta DE DIMENSIUNE num_labels x (n+1), UNDE
% n ESTE NR DE PIXELI DIN FIECARE IMAGINE DE TRAINING (SI ADAUG SI x0 = 1);
% FIECARE LINIE j DIN all_theta INCLUDE PARAMETRII (SAU WEIGHTS) NECESARE PENTRU
% A STABILI DACA IMAGINEA CURENTA POATE FI CLASIFICATA CU LABEL-UL j SAU NU!!!
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

% ANTRENEZ MODELUL PENTRU FIECARE LABEL IN PARTE!!!
for c = 1:num_labels,
  % VREAU SA OBTIN PARAMETRII theta (VECTOR!!!) PENTRU A CLASIFICA IMAGINILE
  % DE TRAINING CA APARTINAND LABEL-ULUI CURENT (c) SAU NU!!!
  initial_theta = zeros(n + 1, 1);
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  % AM OBTINUT PARAMETRII PENTRU LABEL-UL c, II ADAUG IN MATRICEA all_theta,
  % CARE CONTINE PARAMETRII PENTRU FIECARE LABEL IN PARTE (FIECARE LINIE CONTINE
  % PARAMETRII PENTRU UN ANUMIT LABEL
  # FOLOSESC UN LOGICAL ARRAY (y == c) - THE RESULTING VECTOR INCLUDES 1 AT THE
  # INDEX OF ALL ELEMENTS IN y EQUAL TO c, AND 0 FOR ALL OTHER ELEMENTS.
  all_theta(c, :) = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                 initial_theta, options);
endfor



% =========================================================================


end
