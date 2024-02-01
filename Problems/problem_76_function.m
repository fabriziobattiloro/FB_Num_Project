function Fx = problem_76_function(X)
% Function for computing the value of F(x) at point X where F(x) is the one
% reported in problem 76 in [1]
% INPUTS:
% X: n by 1 vector representing a point in the n-dimensional space
% OUTPUTS:
% Fx: the value of the function at X

    n = length(X);
    Fx = (X(n) - X(1).^2 / 10) .^ 2;
    for i = 1:(n-1)
        Fx = Fx + (X(i) - X(i+1).^2 / 10).^2;
    end
    Fx = Fx/2;
end