function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% X is a 5000 x 401 matrix (after adding a column of 1s at the beginning);
% Theta1 is a 25 x 401 matrix mapping from the input layer to the hidden layer.
% Theta2 is a 10 x 26 matrix mapping from the hidden layer to the output layer,
% which is a column vector containing the propabilities that a given image should
% be classified in each of the 10 labels.

% Each line in X is the input layer for the respective image (each line in X 
% contains the 400 pixels of an image we are trying to identify);

% In order to make a prediction for image i (that is, the image whose pixels are
% represented by the values in line i of matrix X), I compute the hidden layer, 
% by multiplying Theta1 (containing the parameters for each element in the 
% hidden layer) with Xi transposed. The sigmoid of the result is a column vector 
% a2(i).

% By multiplying Theta1 with X', and getting the sigmoid of the result, I get a 
% 25 x 5000 matrix, in which each column is the hidden layer for each of the 
% 5000 images in matrix X; (matrix A2 is composed of 5000 column vectors a2(i),
% i = 1...5000)
A2 = sigmoid(Theta1 * X');

% add a first line of 1s in A2 - each vector a2(i) should have 1 as the first
% element.
A2 = [ones(1, size(A2, 2)); A2];


% Theta2 x A2 is a 10 x 5000 matrix. The sigmoid of this matrix represents the 
% output layer.

% I transpose the result, so that the propabilities related to each picture are
% presented in a line in A3, and each column is one of the 10 possible labels.

% Therefore, each line i in A3 is a line vector containing the probabilities that
% image i should be classified in each of the labels in the columns.
A3 = (sigmoid(Theta2 * A2))';

% APELATA IN ACEST FEL, FUNCTIA max RETURNEAZA ELEMNTUL MAXIM (CEA MAI MARE 
% PROBABILITATE DINTRE TOATE LABEL-URILE) DIN FIECARE RAND, IMPREUNA CU 
% INDEXUL LA CARE SE GASESTE ACEST ELEMENT MAXIM. PE MINE MA INTERESEAZA
% DOAR INDEX-UL, CARE INDICA LABEL-UL CU CEA MAI MARE PROBABILITATE PT IMAGINEA i 
[~, p] = max(A3, [], 2);



% =========================================================================


end
