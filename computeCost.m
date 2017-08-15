function J = computeCost(X, y, theta, lambda)

%   Initialization
m = size(X, 1);  %   number of data

%   Compute Cost
J = (1 / m) * sum((-y) .* log(sigmoid(X * theta)) - (1 - y) .* log(1 - sigmoid(X * theta))) + (lambda / (2 * m)) * sum(theta(2:end) .^ 2);

end
