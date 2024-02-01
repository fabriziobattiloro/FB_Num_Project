function gradFx = problem_25_grad(X)
% Function for computing the gradient vector in a specified point of the
% extended rosenbrock
% INPUTS:
% X: n by 1 vector representing a point in the n-dimensional space
% OUTPUTS:
% gradFx: n by 1 vector which represent the gradient at X

grad_odd = @(x, k) 200 * x(k) .^ 3 - 200 .* x(k) .* x(k+1) + x(k) - 1;
grad_even = @(x, k) 100 * x(k) - 100 * x(k-1) .^ 2;
n = length(X);
gradFx = zeros(n, 1);
for i = 1:n
    if mod(i, 2) == 1
        gradFx(i) = grad_odd(X, i);
    else
        gradFx(i) = grad_even(X, i);
    end
end
end