function [xk, fk, gradfk_norm, k, xseq, btseq] = ...
    fletcher_reeves_method(x0, f, gradf, alpha0, ...
    kmax, tolgrad, c1, rho, btmax)

%
% [xk, fk, gradfk_norm, k, xseq, btseq] = ...
    % steepest_desc_bcktrck(x0, f, gradf, alpha0, ...
    % kmax, tolgrad, c1, rho, btmax)
%
% Function that performs the steepest descent optimization method, using
% backtracking strategy for the steplength selection.
%
% INPUTS:
% x0 = n-dimensional column vector;
% f = function handle that describes a function R^n->R;
% gradf = function handle that describes the gradient of f;
% alpha0 = the starting value for the step length, before backtracking;
% kmax = maximum number of iterations permitted;
% tolgrad = value used as stopping criterion w.r.t. the norm of the
% gradient;
% c1 = ï»¿the factor of the Armijo condition that must be a scalar in (0,1);
% rho = ï»¿fixed factor, lesser than 1, used for reducing alpha0;
% btmax = ï»¿maximum number of steps for updating alpha during the 
% backtracking strategy.
%
% OUTPUTS:
% xk = the last x computed by the function;
% fk = the value f(xk);
% gradfk_norm = value of the norm of gradf(xk)
% k = index of the last iteration performed
% xseq = n-by-k matrix where the columns are the xk computed during the 
% iterations
% btseq = 1-by-k vector where elements are the number of backtracking
% iterations at each optimization step.
%

% Function handle for the armijo condition
farmijo = @(fk, alpha, gradfk, pk) ...
    fk + c1 * alpha * gradfk' * pk;

xseq = zeros(length(x0), kmax);
btseq = zeros(1, kmax);

xk = x0;
fk = f(xk);
gradfk = gradf(xk);
k = 0;
gradfk_norm = norm(gradfk);
pk = -gradf(xk);

while k < kmax && gradfk_norm >= tolgrad
    
    % Reset the value of alpha
    alpha = alpha0;
    
    % Compute the candidate new xk
    xnew = xk + alpha * pk;
    % Compute the value of f in the candidate new xk
    fnew = f(xnew);
    
    bt = 0;
    % Backtracking strategy: 
    % 2nd condition is the Armijo condition not satisfied
    while bt < btmax && fnew > farmijo(fk, alpha, gradfk, pk)
        % Reduce the value of alpha
        alpha = rho * alpha;
        % Update xnew and fnew w.r.t. the reduced alpha
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        
        % Increase the counter by one
        bt = bt + 1;
        
    end
    
    gradfknew = gradf(xnew);
    beta = norm(gradfknew)/gradfk_norm;
    pk = -gradfknew + beta*pk;
    
    % Update xk, fk, gradfk_norm
    xk = xnew;
    fk = fnew;
    gradfk = gradfknew;
    gradfk_norm = norm(gradfk);
    
    % Increase the step by one
    k = k + 1;
    
    % Store current xk in xseq
    xseq(:, k) = xk;
    % Store bt iterations in btseq
    btseq(k) = bt;
end

% "Cut" xseq and btseq to the correct size
xseq = xseq(:, 1:k);
btseq = btseq(1:k);

end
