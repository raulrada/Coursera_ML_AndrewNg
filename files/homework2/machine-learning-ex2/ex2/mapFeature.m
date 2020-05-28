function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

% the output is a matrix of form:
% [1 x11 x12 x11^2 x11*x12 x12^2 x11^3 x11^2*x12 x11*x12^2 x12^3 ... x11*x12^5 x12^6
%  .................................................................................
%  1 xj1 xj2 xj1^2 xj1*xj2 xj2^2 ....................................xj1*xj2^5 xj2^6
%  .................................................................................
%  1 xm1 xm2 xm1^2 xm1*xm2 xm2^2 ....................................xm1*xm2^5 xm2^6]

degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end