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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%transform y to vectors
y_vectors = zeros(size(y,1), num_labels);
for i = 1 : m
   y_vectors(i,y(i)) = 1;
end

%Part 1

%计算h(x)
h = zeros(size(X, 1), 1);
a1 = [ones(m,1) X];
a2 = sigmoid(Theta1*a1');

k = size(a2,2);
a2 = [ones(1,k); a2];

a3 = sigmoid(Theta2*a2);
h=a3';

%计算J
for i = 1 : m
    temp = 0;
    for j = 1 : num_labels
        temp = temp + (-y_vectors(i,j)*log(h(i,j)) - (1-y_vectors(i,j))*log(1-h(i,j)));
    end
    J = J + temp;
end

J = J / m;

%计算正则化部分
reg = 0;
for i = 1:hidden_layer_size
    for j = 2:input_layer_size+1
        reg = reg + Theta1(i,j)^2;
    end
end

for i = 1:num_labels
    for j = 2:hidden_layer_size+1
        reg = reg + Theta2(i,j)^2;
    end
end

J = J + reg*lambda/2/m;

%part 2
clear a1 a2 a3 z1 z2 z3;
D_delta1 = zeros(size(Theta1));
D_delta2 = zeros(size(Theta2));

for i = 1:m
   
    %step 1 forward
    a1 = [1;X(i,:)'];
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    %step 2 layer 3
    delta3 = a3 - y_vectors(i,:)';
    
    %step 3 layer 2
    z2 = [1;z2];
    delta2 = Theta2'*delta3.*sigmoidGradient(z2);
    
    %step 4 added
    delta2 = delta2(2:end);
    D_delta2 = D_delta2 + delta3*a2';
    D_delta1 = D_delta1 + delta2*a1';
    
end

Theta2_grad = D_delta2/m + lambda/m * [zeros(num_labels,1) Theta2(:, 2:end)];
Theta1_grad = D_delta1/m + lambda/m * [zeros(hidden_layer_size,1) Theta1(:, 2:end)];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
