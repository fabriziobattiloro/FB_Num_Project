function gradFx = problem_81_grad(X)
% Function for computing the gradient vector at a specified point of the
% function reported in problem 81 in [1]
% INPUTS:
% X: n by 1 vector representing a point in the n-dimensional space
% OUTPUTS:
% gradFx: n by 1 vector which represent the gradient at X
    n = length(X);
    gradFx = zeros(n, 1);
    gradFx(1) = 4 * X(1).^3 + 2 * X(1) * log(X(2)) - 4 * X(1);
    gradFx(n) = (X(n-1).^2 + log(X(n)) - 1) / X(n);
    for i = 2:(n-1)
        gradFx(i) = (X(i-1).^2 + log(X(i)) - 1) / X(i) + 2 * X(i).^3 + 2*X(i) * log(X(i+1)) - 2 * X(i);
    end
end