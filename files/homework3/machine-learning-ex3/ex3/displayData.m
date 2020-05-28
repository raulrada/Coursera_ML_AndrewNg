function [h, display_array] = displayData(X, example_width)
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.

% Set example_width automatically if not passed in

% ~exist('example_width', 'var') verifies not(there is a variable named example_width)
% it looks like the code makes sure the program doesn't crash in case the 
% example_width is not provided
% since ex3.m does not provide example_width, it is computed hereinafter!
if ~exist('example_width', 'var') || isempty(example_width) 
  % since images have an equal width and height, the width is the square root
  % of the number of columns (since each row contains all pixels for a given image,
  % as a row vector; each line contains 20 x 20 pixels
  example_width = round(sqrt(size(X, 2)));
end

% Gray Image
colormap(gray);

% Compute rows, cols
% here, X is actually sel variable from ex3.m, having size 100 x 400
[m n] = size(X);
% having the pixels for one image as a row vector (one row in the matrix), and 
% knowing the width of each example, the height must be the total number of pixels
% (i.e., the number of columns in the matrix) divided by width, because the 
% number of columns equals example_height x example_width
example_height = (n / example_width);

% Compute number of items to display
% we want to show the images in a square grid (i.e.: 10 x 10 images).
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% Between images padding
pad = 1;

% Setup blank display
% it results in a 211 x 211 matrix with values -1
% pad = 1; display_rows = 10; example_height = 20; display_cols = 10; example_width = 20
% -1 represents black - initialize the matrix to all-black pixels
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));
                       
% Copy each example into a patch on the display array
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch
		max_val = max(abs(X(curr_ex, :)));
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Display Image
h = imagesc(display_array, [-1 1]);

% Do not show axis
axis image off

drawnow;

end
