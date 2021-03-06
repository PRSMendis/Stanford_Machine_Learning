function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%%ITERATIVE APPROACH
% for i= 1:m
%     %Linear Regression
% %     J = J + (X(i,:) * theta - y(i)) ^ 2;
%     sig = sigmoid(X(i,:)*theta);
%    J =J + (-y(i) * log(sig)) - (1-y(i)*log(sig));
% end
% J = J / m;
%% VECTORIZED IMPLEMENTATION: 

% h = X * theta ; 
h = sigmoid(X * theta) ; 
% J = 1/m * (-(y')*log(h)) - ((1-y)'*log(1-h));
% J = 1/m * sum(-y.*log(h) - ((1-y).*log(1-h)));
J = (1 / m) * (log(h)' * -y - log(1 - h)' * (1 - y));

%gradient descent
% alpha = 0.01;
% theta = ((alpha/ m) * (sigmoid(h) - y));
grad = (X' * (h-y))/m;


% htheta = sigmoid(X * theta);
% J = 1/m * sum(-y .* log(htheta) - (1 - y) .* log(1 - htheta));
% % calculate grads
% grad = (X'*(htheta - y))/m;





% =============================================================




