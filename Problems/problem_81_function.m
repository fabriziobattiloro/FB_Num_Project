function Fx = problem_81_function(X)
% Function for computing the value of F(x) at point X where F(x) is the one
% reported in problem 81 in [1]
% INPUTS:
% X: n by 1 vector representing a point in the n-dimensional space
% OUTPUTS:
% Fx: the value of the function at X
    n = length(X);
    Fx = (X(1) .^2 - 1).^2;
    for i = 2:n
        Fx = Fx + (X(i-1).^2 + log(X(i)) - 1).^2;
    end
    Fx = Fx/2;
end