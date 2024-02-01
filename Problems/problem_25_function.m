function Fx = problem_25_function(X)
% Function for computing the value F(x) where F is the extended rosenbrock
%
% INPUTS:
% X: n by 1 vector representing a point in the n-dimensional space
% OUTPUTS:
% Fx: the value of the extended rosenbrock function at X

f_k_odd = @(x, k) 100 * (x(k).^2 - x(k+1)) .^ 2;
f_k_even = @(x, k) (x(k-1) - 1) .^ 2;
n = length(X);
Fx = 0;
for i=1:n
    if mod(i, 2) == 1
        Fx = Fx + f_k_odd(X, i);
    else
        Fx = Fx + f_k_even(X, i);
    end
end
Fx = Fx ./ 2;
end