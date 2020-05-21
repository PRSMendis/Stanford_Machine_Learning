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
Yd = Y';

%Set accumulator to calculate the overall cost
DELTA = 0;
for t = 1:m 
    
%     Step 1
    
%     Set the input layer's values  to the t-th training example . Perform a feedforward pass (Figure 2),
%     computing the activations  for layers 2 and 3. Note that you need to add a  term to ensure that the vectors of
%     activations for layers  and  also include the bias unit.
%     In MATLAB, if a_1 is a column vector, adding one corresponds to a_1 = [1; a_1].

    %Load in first training example
    a_1 = X(t,:) ; 
    %Add bias unit - not needed because bias unit already added in part 1
%     a_1 = [1; a_1];
    a_2 = sigmoid(a_1 * Theta1')';
%     a_2 = a_1 * Theta1';
    a_2 = [1; a_2] ;
%     a_3 = a_2 * Theta2' ;
    a_3 = sigmoid(Theta2 * a_2);
    
    
    %Step 2
    
%     For each output unit  in layer 3 (the output layer), set delta3 = (a3 - y)  where  indicates whether the current training example belongs to class 
%     , or if it belongs to a different class .You may find logical arrays helpful for this task (explained in the previous programming exercise).
%     For the hidden layer , set 

    d_3 = a_3 - Yd(:,t);
    
    %Step 3
%     For the hidden layer l = 2, set delta_2 = theta_2' * delta_3 . *
%     g(z^2))

    d_2 = (Theta2' * d_3) .*  a_2;
    
    %Step 4
%     Accumulate the gradient from this example using the following
%     formula: DELTA_L = DELTA_L = delta_(L+1) * (a_L)'
%     . Note that you should skip or remove delta_0^(2) . In MATLAB, removing  corresponds to delta_2 = delta_2(2:end).

% DELTA = DELTA + (d_3 * a_2');
% DELTA = DELTA + sum(sum((d_3 * a_2')));
% DELTA = DELTA + sum(sum((d_3 * a_2')))' + sum((d_2(2:end) * a_1));
DELTA = DELTA + sum(sum(sum((d_3 * a_2')))' + sum((d_2(2:end) * a_1)));
% DELTA = sum(DELTA);
    
    
    
    
    
end

% Step 5


% Obtain the (unregularized) gradient for the neural network cost function
% by dividing the accumulated gradients by  1/m



D_ij = 1/m * DELTA;
Theta1_grad = 1/m * (DELTA + lambda * Theta1) ;
Theta2_grad = 1/m * (DELTA + lambda * Theta2) ;











%%
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
    