function [gradfx] = findiff_grad(F, x, k, type)
%
% function [gradf] = findiff_grad(f, x, h, type)
%
% Function that approximate the gradient of f in x (column vector) with the
% finite difference (forward/centered) method.
%
% INPUTS:
% f = function handle that describes a function R^n->R;
% x = n-dimensional column vector;
% h = the h used for the finite difference computation of gradf
% type = 'fw' or 'c' for choosing the forward/centered finite difference
% computation of the gradient.
%
% OUTPUTS:
% gradfx = column vector (same size of x) corresponding to the approximation
% of the gradient of f in x.

n = length(x);
gradfx=zeros(n,1);

h=10^(-k)*norm(x);

for i = 1:n
    xh = x;
    xh(i) = x(i) + h;
    switch type
         case 'fw'
            gradfx(i)=(F(xh)-F(x))/h;
         case 'c'
             xh_minus = x;
             xh_minus(i) = x(i) - h;
             gradfx(i)=(F(xh)-F(xh_minus))/(2*h);
         otherwise %we do forward one
             gradfx(i)=(F(xh)-F(x))/h;
    end
end
end
