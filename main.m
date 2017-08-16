%%  Pen-Based Recognition of Handwritten Digits By Regularized Multivariate Multiclass Logistic Regression
%   Programed by Daniel H. Leung 15/08/2017 (DD/MM/YYYY)
%   For detail, please refer to the comments in codes and README.md

%%  Initialization

clear;
close all;
clc;

%%  Load Data

X = load('in_train.txt');
y = load('out_train.txt');
X_test = load('in_test.txt');
y_test = load('out_test.txt');
num_label = 10;
lambda = 0.1;
epoch = 5000;
alpha = 4.0;

%%  Feature Scaling

[X, mu, stddev] = featureScaling(X);    %   get the scaled X, mean of each feature, standard deviation of each feature

%%  One-vs-All Classification Training

[all_theta, J_History] = oneVSall(X, y, num_label, lambda, epoch, alpha);

%%  Plot graphs and display all thetas

%   Plot all graphs of Costs - Num of Iterations
for i = 0:(num_label - 1)
    title = strcat('Cost of Classifier of Digit', int2str(i), ' - Num of Iterations');
    figure('Name', title);
    plot(1:numel(J_History(:, i + 1)), J_History(:, i + 1), '-b', 'LineWidth', 2);
    xlabel('Num of Iterations');
    ylabel(strcat('Cost of Classifier of Digit', int2str(i)));
end

%   Display all_theta
for i = 0:(num_label - 1)
    fprintf('Theta for Logistic Classifier of Digit %d\n\n', i);
    fprintf('%f\n', all_theta(:, i + 1));
    fprintf('\n\n');
end

%%  Predictions to testcase

for i = 1:size(X_test, 2)
   X_test(:, i) = (X_test(:, i) - mu(i)) / stddev(i);
end
X_test = [ones(size(X_test, 1), 1), X_test];
[temp, predict] = max(sigmoid(X_test * all_theta), [], 2);  %   find out index of that max value and save in predict
predict = predict - 1;  %   matlab's indexing starts from 1, but my code considers index 1 as digit 0
fprintf('Tested by test dataset.\n');
fprintf('Accuracy: %f%%\n', mean(double(predict == y_test)) * 100);
