function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% go over all examples in matrix X
for i = 1:size(X, 1)
  % for each example i in X, compute matrix differences (same size as centroids 
  % matrix) - with each line representing the element-wise difference between 
  % current example i in X and each of the K centroids (i.e.: line j in 
  % difference is the element-wise difference between example i in matrix X and
  % centroid j, j = 1...k)
  differences = X(i,:) - centroids;
  
  % Compute the norm of each vector (xi - centroid(j)). Square the element-wise
  % differences between xi and centroid(j) and add the differences (sum after 2nd
  % dimension). Each line in distances matrix is the squared norm of difference
  % vectors of the form xi - centroid(j).
  % The norm of the difference vector are the square roots of the afore mentioned
  % sums of squared differences, but, since we are only interested in the index
  % of the closest centroid (that is, centroid(j) for which the sum of the 
  % squared differences is the lowest), we do not need to perform an additional
  % computation and take the square root(euclidian geometry) in order to get 
  % the actual norms of the difference vectors.
  squared_distances = sum(differences .^ 2, 2);
  
  % get only the index of the minimum value in squared_distances, and disregard
  % the actual minimum value
  [~, idx(i)] = min(squared_distances);
endfor




% =============================================================

end

