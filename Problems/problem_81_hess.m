function HessFx = problem_81_hess(X)
% Function for computing the hessian matrix at a given point of the
% function reported in problem 81 in [1]
% INPUTS:
% X: n by 1 vector representing a point in the n-dimensional space
% OUTPUTS:
% HessFx: a n-by-n sparse tridiagonal matrix representing the hessian
% matrix at point X
    n = length(X);
    max_nz = 3*n - 2;
    row = zeros(max_nz, 1);
    column = zeros(max_nz, 1);
    values = zeros(max_nz, 1);
    nz = 1;
    row(nz) = 1;
    column(nz) = 1;
    values(nz) = 12 * X(1) + 2*log(X(2)) - 4;
    for i = 2:(n-1)
        nz = nz + 1;
        row(nz) = i;
        column(nz) = i;
        values(nz) = (-log(X(i)) + X(i-1).^2 - 2) / X(i).^2 + 6 * X(i).^2 + 2 * log(X(i+1)) - 2;
        tmp = 2 * X(i-1) / X(i);
        nz = nz + 1;
        row(nz) = i;
        column(nz) = i-1;
        values(nz) = tmp;
        nz = nz + 1;
        row(nz) = i-1;
        column(nz) = i;
        values(nz) = tmp;
    end
    nz = nz + 1;
    row(nz) = n;
    column(nz) = n;
    values(nz) = (-log(X(n)) + X(n-1).^2 - 2) / X(n);
    tmp = 2*X(n-1) / X(n);
    nz = nz + 1;
    row(nz) = n;
    column(nz) = n-1;
    values(nz) = tmp;
    nz = nz + 1;
    row(nz) = n-1;
    column(nz) = n;
    values(nz) = tmp;
    HessFx = sparse(row, column, values, n, n);

    % check if the hessian is SPD
    b = trace(HessFx);
    if b <= 0
        warning('The trace of the hessian is non-positive');
    end
end