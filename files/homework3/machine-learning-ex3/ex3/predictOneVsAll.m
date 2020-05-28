function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% VREAU MATRICE DE DIMENSIUNEA m * number of labels, IN CARE FIECARE RAND i CONTINE
% PROBABILITATEA CA IMAGINEA i DIN TRAINING SET SA FIE DIN FIECARE LABEL DE 
% LA 1 LA number of labels;
% FIECARE VECTOR  xi (CORESPUNZATOR PIXELILOR POZEI i) IL INMULTESC CU theta1,
% theta2, ..., theta indice number of labels, SI CALCULEZ sigmoid(fiecare produs);
% ASTFEL, FIECARE RAND DIN prediction_matrix CONTINE PROBABILITATEA CA POZA i SA
% APARTINA FIECARUIA DIN LABEL-URILE DE LA 1 LA number of labels!!!
prediction_matrix = sigmoid(X * all_theta');

% APELATA IN ACEST FEL, FUNCTIA max RETURNEAZA ELEMNTUL MAXIM (CEA MAI MARE 
% PROBABILITATE DINTRE TOATE LABEL-URILE) DIN FIECARE RAND, % IMPREUNA CU 
% INDEXUL LA CARE SE GASESTE ACEST ELEMENT MAXIM. PE MINE MA INTERESEAZA
% DOAR INDEX-UL, CARE INDICA LABEL-UL CU CEA MAI MARE PROBABILITATE PT IMAGINEA i 
[~, p] = max(prediction_matrix, [], 2);

% =========================================================================


end
