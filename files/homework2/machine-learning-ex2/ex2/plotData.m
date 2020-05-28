function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% find indices of elements 1 and 0 in y vector
pos = find (y==1);
neg = find (y==0);

% We have the index of elements in vector y with value 1. We represent points
% x-coordinates equal to the exam 1 results (column 1 in X) corresponding to the
% same index element in y (with y = 1) and y-coordinates equal to exam 2 
% results (column 2 in X, taking only values with the same index as those equal to 1 in y)
% corresponding to the same index element in y
% parameter '+' , '*' or similar plots values as points, instead of as lines!!!
% 'k+' plots values as plus sign, with color black (k)
% make the line thicker ('LineWidth', 2)
plot(X(pos,1), X(pos,2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);

% plot exam 1 and 2 results for rejected candidates (y = 0), with 'o' on graph;
% outline of the 'o' is drawn with black; the 'o' is filled in yellow 
% ('MarkerFaceColor', 'y')
plot(X(neg,1), X(neg,2), 'ok', 'MarkerSize', 7, 'MarkerFaceColor', 'y');





% =========================================================================



hold off;

end
