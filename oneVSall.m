function [all_theta, J_History] = oneVSall(X, y, num_label, lambda, epoch, alpha)

%   Initialization
m = size(X, 1);  %   number of data
n = size(X, 2); %   number of features
all_theta = zeros(n + 1, num_label);    %   thetas according the num_label
X = [ones(m, 1), X];    %   adding intercept terms to X
J_History = zeros(epoch, num_label);

%   Train one vs all logistic regression of the amount of num_label
%   Save corresponding theta and cost history for every classifier of each digit
for i = 0:(num_label - 1)
    fprintf('Training Logistic Classifier of Digit %d...\n', i);
    [all_theta(:, i + 1), J_History(:, i + 1)] = gradientDescent(X, y == i, lambda, epoch, alpha);
end
fprintf('\n');

end