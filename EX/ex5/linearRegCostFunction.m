function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h = X * theta ;
% J = 1/2m * sum(h-y) ; 

%Calculate penalty
% p = (sum(theta)).^2;

% Remember to not regularize the first term of theta
p = (sum(theta.^2)-theta(1,1)^2);


% reg = 0.0415, * 24 =  0.9954
% J = ((1/(2*m)) * (sum(h-y).^2)) + ((lambda/(2*m)) * p) ;
J = (1/(2*m)) * (sum((h-y).^2)) + ((lambda/(2*m)) * p);


%Calculating grad
grad = 1/m * (X' * (h-y)); 
thetaT = theta;
thetaT(1,1) = 0;
grad = grad + (lambda* thetaT)/m ;











% =========================================================================

grad = grad(:);

end
