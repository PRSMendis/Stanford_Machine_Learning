function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
% k =1:m;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % use partial derivates to updates values of theta...
    
%     theta = theta - alpha * computeCost(X, y, theta);
%     t1 = theta(0) - alpha * (X(i,:) * theta - y(i))
%     t2 = theta(0) - alpha * ((X(i,:) * theta - y(i))*(X(i,:))
    h=X*theta;
    g=h-y;
    delt=(1/m)*(alpha*(X'*g));

    theta=theta-delt;








    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
