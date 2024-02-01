function HessFx = problem_25_hess(X)
% Function for computing the hessian matrix of extended rosenbrock at a
% given point X
% INPUTS:
% X: n by 1 vector representing a point in the n-dimensional space
% OUTPUTS:
% HessFx: a n-by-n sparse tridiagonal matrix representing the hessian
% matrix at point X

    n = length(X);
    max_nz = 2*n - 1;
    row = zeros(max_nz, 1);
    column = zeros(max_nz, 1);
    values = zeros(max_nz, 1);
    nz = 0;
    for i=1:(n-1)
        if mod(i,2) == 1
           nz = nz+1;
           row(nz) = i;
           column(nz) = i;
           values(nz) = 600 * X(i).^2 - 200 * X(i+1) + 1;
           tmp = -200 * X(i);
           nz = nz + 1;
           row(nz) = i;
           column(nz) = i+1;
           values(nz) = tmp;
           nz = nz + 1;
           row(nz) = i+1;
           column(nz) = i;
           values(nz) = tmp;
        else
            nz = nz + 1;
            row(nz) = i;
            column(nz) = i;
            values(nz) = 100;
        end
    end
    nz = nz + 1;
    row(nz) = n;
    column(nz) = n;
    values(nz) = 100;
    HessFx = sparse(row, column, values, n, n);
end