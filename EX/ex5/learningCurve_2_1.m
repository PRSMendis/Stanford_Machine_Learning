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
x_size = size(X);
divider = 0.6;

i = floor(x_size(1) * divider);


%Calculating 
train_set = X(1:i, :);
T_X = train_set;
train_rslt = y(1:i);
T_y = train_rslt;

V_X = X(i:end, :);
V_y = y(i:end);


% lambda = 0;
% [theta] = trainLinearReg([ones(m, 1) X], y, lambda);

%Training theta values on training set
[theta] = trainLinearReg(T_X, T_y, lambda);

%Acquiring cost of training set
[J, grad] = linearRegCostFunction(T_X, T_y, theta, lambda);

error_train = J;


% error_val = 9999999999999999999;
error_val = linearRegCostFunction(V_X, V_y, theta, 0);
for lambda = 1:10
%     pop = lambda
    [temp_error_val, grad] = linearRegCostFunction(V_X, V_y, theta, lambda);
    if temp_error_val <  error_val
        error_val = temp_error_val;
    end
    
    
    
end



% -------------------------------------------------------------

% =========================================================================

end
