function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% Go over each centroid
for i = 1:K
  % get a logical vector, with values 1 at indexes where idx has the value of 
  % the current centroid. For instance, if idx(j) == current centroid, then
  % is_curr_centroid(j) == 1, otherwise the value of is_curr_centroid == 0
  is_curr_centroid = (idx == i);
  
  % The number of elements in X allocated to current centroid is equal to the
  % number of non-zero elements in logical vector is_curr_centroid
  num_elem_curr_centroid = sum(is_curr_centroid);
  
  % Perform element-wise multiplications of each value in X with the logical
  % vector is_curr_centroid, so that the only remaining non-zero elements in X 
  % are the coordinates of the examples classified in the group with the current 
  % centroid
  X_curr_centroid = X .* is_curr_centroid;
  
  % update the value of the current centroid, to be the coordinate-wise average
  % of the examples in X allocated to the group with the current centroid.
  centroids(i, :) = sum(X_curr_centroid) / num_elem_curr_centroid;
endfor






% =============================================================


end

