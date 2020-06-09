function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
% j = 1/m;
% j = j * (-y) * log(h)


%Cost of non-neural network
%h = sigmoid(X * theta) ; 
%J = ((1 / m) * (log(h)' * -y - log(1 - h)' * (1 - y))) + ((lambda/(2*m)) * sum(theta(2:length(theta)).^2 ));
% Prediction of neural network
% X = [ones(m, 1) X];
% C = sigmoid(X * Theta1');
% n = size(C,1);
% C = [ones(n, 1) C];
% D = sigmoid(C * Theta2');
% [val,p] = max(D,[],2);


%use indexing of the identity matrix to essentially get all different
%classificaitons, in this case - 10

% eye_m = eye(num_labels);
% y = eye_m

I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end





%First compute the outputs of the 1st layer
X = [ones(m, 1) X];
h1 = sigmoid(X * Theta1');
h1 = [ones(m,1) h1];
%Compute output of the 2nd layer
h2 = sigmoid(h1 * Theta2');
h = h2;
%Find the cost 

%Consider that the cost is based on the 10 different classifiers?? I think
% J = ((1 / m) * (log(h2)' * -y - log(1 - h2)' * (1 - y))) + ((lambda/(2*m)) * sum(Theta2(2:length(Theta2)).^2 ));
% J = ((1 / m) * (log(h2)' * -y - log(1 - h2)' * (1 - y))) + ((lambda/(2*m)) * sum(Theta2(2:length(Theta2)).^2 ));

% J = sum(J);

% for j = 1:length(num_labels)
%     J = J + ((1 / m) * (log(h2)' * -Y(:,j) - log(1 - h2)' * (1 - Y(:,j)))) + ((lambda/(2*m)) * sum(Theta2(2:length(Theta2)).^2 ));
% end

p = sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2));

% J = (-Y).*log(h2) - (1-Y).*log(1-h2) ;

% J = sum(sum((-Y).*log(h) - (1-Y).*log(1-h), 2))/m + lambda*p/(2*m);
J = sum(sum((-Y).*log(h) - (1-Y).*log(1-h), 2))/m + lambda*p/(2*m);

% c1 = eye_m(:,1);
% c2 = eye_M(:,2);%...


%%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% In detail, here is the backpropagation algorithm (also depicted in Figure 3). 
% You should implement steps 1 to 4 in a loop that processes one example at a time. 
% Concretely, you should implement a for loop for t = 1:m and place steps 1-4 below inside the for loop, 
% with the  iteration performing the calculation on the  training example . Step 5 will divide the accumulated gradients
% by  to obtain the gradients for the neural network cost function.

%Old Grad calculation
% grad = (X' * (h-y))/m;
% temp = theta;
% temp(1) = 0;
% grad = grad + (lambda/m * temp) ;
% calculate sigmas
% sigma3 = a3.-Y;
sigma2 = (sigma3*Theta2).*sigmoidGradient([ones(size(z2, 1), 1) z2]);
sigma2 = sigma2(:, 2:end);

% accumulate gradients
delta_1 = (sigma2'*a1);
delta_2 = (sigma3'*a2);

% calculate regularized gradient
p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
Theta1_grad = delta_1./m + p1;
Theta2_grad = delta_2./m + p2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
