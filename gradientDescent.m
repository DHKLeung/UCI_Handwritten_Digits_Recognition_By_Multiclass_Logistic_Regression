function [theta, J_History] = gradientDescent(X, y, lambda, epoch, alpha)

%   Initialization
m = size(X, 1);  %   number of data
n = size(X, 2); %   number of features
theta = zeros(n, 1);
J_History = zeros(epoch, 1);

%   Gradient Descent
for i = 1:epoch
    temp = theta;
    temp(1) = 0;
    grad = (1 / m) * X' * (sigmoid(X * theta) - y) + ((lambda / m) * temp); %   Compute gradient
    theta = theta - alpha .* grad;  %   Update theta
    
    %   Save the cost J in every iteration
    J_History(i) = computeCost(X, y, theta, lambda);
end

end