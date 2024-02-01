function [Hessfx] = findiff_hess(f, x, exp)
%
% [Hessf] = findiff_Hess(f, x, h)
%
% Function that approximate the Hessian of f in x (column vector) with the
% finite difference method.
% ATTENTION: we assume a regular f (C^2) s.t. Hessf is symmetric.
%
% INPUTS:
% f = function handle that describes a function R^n->R;
% x = n-dimensional column vector;
% h = the h used for the finite difference computation of Hessf
%
% OUTPUTS:
% Hessfx = n-by-n matrix corresponding to the approximation of the Hessian 
% of f in x.

n = length(x);
Hessfx = zeros(n);
h = 10^(-exp)*norm(x);

for j=1:n
    % Elements on the diagonal
    xh_plus = x;
    xh_minus = x;
    xh_plus(j) = xh_plus(j) + h;
    xh_minus(j) = xh_minus(j) - h;
    Hessfx(j,j) = (f(xh_plus) - 2*f(x) + f(xh_minus))/(h^2);
    % ALTERNATIVELY (no xh_plus and xh_minus)
    % Hessf(j,j) = (f([x(1:j-1); x(j)+h; x(j+1:end)]) - 2*f(x) + ...
    %     f([x(1:j-1); x(j)-h; x(j+1:end)]))/(h^2);
    for i=j+1:n
        xh_plus_ij = x;
        xh_plus_ij([i, j]) = xh_plus_ij([i, j]) + h;
        xh_plus_i = x;
        xh_plus_i(i) = xh_plus_i(i) + h;
        xh_plus_j = x;
        xh_plus_j(j) = xh_plus_j(j) + h;
        Hessfx(i,j) = (f(xh_plus_ij) - ...
            f(xh_plus_i) - f(xh_plus_j) + f(x))/(h^2);
        
        % ALTERNATIVELY (no xh_plus_i/j)
        % Hessf(i,j) = (f(xh_plus_ij) - ...
        %     f([x(1:i-1); x(i)+h; x(i+1:end)]) - ...
        %     f([x(1:j-1); x(j)+h; x(j+1:end)]) + f(x))/(h^2);
        
        Hessfx(j,i)=Hessfx(i,j);
    end
end


end