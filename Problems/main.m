% Script for minimizing the function in various problems
% INITIALIZATION
close all; clear; clc;
disp('** PROBLEMS: **');
rho = 0.5; 
c = 1e-4; 
kmax = 10000; 
tolgrad = 1e-12;
alpha0 = 1;
btmax = 50;
pcg_maxit = 50;
type = "fw";


%Function handles: UNCOMMENT THE EXERCISE TO MINIMIZE
% f = @(x) problem_25_function(x); % value of the function
% gradf = @(x) problem_25_grad(x); % gradient vector
% Hessf = @(x) problem_25_hess(x); % hessian matrix

f = @(x) problem_81_function(x); % value of the function
gradf = @(x) problem_81_grad(x); % gradient vector
Hessf = @(x) problem_81_hess(x); % hessian matrix

%f = @(x) problem_76_function(x); % value of the function
%gradf = @(x) problem_76_grad(x); % gradient vector
%Hessf = @(x) problem_76_hess(x); % hessian matrix


% The Space Dimension 
n_values = [10^3]; 


for j = 1:length(n_values)
    n = n_values(j);
    methods = strings([2,1]);
    elapsed_times = zeros(2,1) ;
    grad_norm_last = zeros(2,1) ;
    fk_last = zeros(2,1) ;
    k_iterations = zeros(2,1) ;
    mem = 1;
    r = 1;
    
    disp(['SPACE DIMENSION: ' num2str(n, '%.0e')]);

    %---------------- generating starting point

    % FOR EXERCISE 25
    % x0 = ones(n,1);
    % i = 1; 
    % while i<=n
    %    x0(i) = -1.2; 
    %    i = i + 2; 
    % end


  
    x0 = 0.5 * ones(n, 1); % FOR EXERCISE 81


    %x0 = 2 * ones(n, 1); % FOR EXERCISE 76
    

    %---------------------------------------------------------------------
    %-------------- Newton Method with backtracking 
    tic;

    [~, fk, gradfk_norm, k, ~, ~] = newton_general_with_bt(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c, rho, btmax);

    elapsed_time = toc;

    %--------------- Collecting results 

    methods(r) = 'Newton exact' ;% 1st column method name
    elapsed_times(r) = elapsed_time ; % second column elapsed time
    grad_norm_last(r) = gradfk_norm(end) ; % 3rd column grad_norm last
    fk_last(r) = fk(end) ; % 4th column function value last
    k_iterations(r) = k ; % 5th column k iteration
    r = r+1 ;

    %---------------------------------------------------------------------
    %--------------------- Steepest Descent Method with backtracking 
     
     tic;
     
     [~, fk, gradfk_norm, k, ~ , btseq] = steepest_descent_with_bt(x0, f, ...
      gradf, alpha0, kmax, tolgrad, c, rho, btmax);

     elapsed_time = toc;

     %--------------- Collecting results 

     methods(r) = 'Steepest Descent' ;% 1st column method name
     elapsed_times(r) = elapsed_time ; % second column elapsed time
     grad_norm_last(r) = gradfk_norm(end) ; % 3rd column grad_norm last
     fk_last(r) = fk(end) ; % 4th column function value last
     k_iterations(r) = k ; % 5th column k iteration
     r=r+1;

     %---------------------------------------------------------------------
     %--------------------- Fletcher Reeves Method
     
     % tic;
     % 
     % [~, fk, gradfk_norm, k, ~ , btseq] = fletcher_reeves_method(x0, f, ...
     %  gradf, alpha0, kmax, tolgrad, c, rho, btmax);
     % 
     % elapsed_time = toc;
     % 
     % %--------------- Collecting results 
     % 
     % methods(r,1) = 'Fletcher Reeves' ;% 1st column method name
     % elapsed_times(r) = elapsed_time ; % second column elapsed time
     % grad_norm_last(r) = gradfk_norm(end) ; % 3rd column grad_norm last
     % fk_last(r) = fk(end) ; % 4th column function value last
     % k_iterations(r) = k ; % 5th column k iteration
     %r = r+1;

     %---------------------------------------------------------------------
     %--------------------- Inexact Newton Method
    
     fterms = @(gradfk,k)min(0.5,norm(gradfk));

     tic;     
     [xk, fk, gradfk_norm, k, xseq, btseq, pcgiterseq] = ...
        innewton_bcktrck(x0, f, gradf, Hessf, kmax, ...
            tolgrad, c, rho, btmax, fterms, pcg_maxit);

     elapsed_time = toc;

     %--------------- Collecting results 

     methods(r,1) = 'Inexact Newton' ;% 1st column method name
     elapsed_times(r) = elapsed_time ; % second column elapsed time
     grad_norm_last(r) = gradfk_norm(end) ; % 3rd column grad_norm last
     fk_last(r) = fk(end) ; % 4th column function value last
     k_iterations(r) = k ; % 5th column k iteration
     r = r+1;


     %---------------------------------------------------------------------
     %--------------------- Modified Newton Method
    
     tic;
     
     [xk, fk, gradfk_norm, k, xseq, btseq] = ...
            modified_newton_method(x0, f, gradf, Hessf, kmax, ...
               tolgrad, c, rho, btmax);

     elapsed_time = toc;

     %--------------- Collecting results 

     methods(r,1) = 'Modified Newton' ;% 1st column method name
     elapsed_times(r) = elapsed_time ; % second column elapsed time
     grad_norm_last(r) = gradfk_norm(end) ; % 3rd column grad_norm last
     fk_last(r) = fk(end) ; % 4th column function value last
     k_iterations(r) = k ; % 5th column k iteration
     r = r+1;


    %----------------------------------------------------------------------
     %--------------------- Nelder Mead Method
     
     % alpha= 1;
     % beta= 0.5;
     % gamma= 2;
     % delta = 0.5;
     % lambda_val = 1;
     % eps1= 10^-3;
     % eps2= 10^-3;
     % tic;
     % 
     % [xmin, Fmin, k] = ...
     %        nelder_mead_method(x0, f, n, kmax, alpha, beta, gamma, delta, lambda_val, eps1, eps2);
     % 
     % elapsed_time = toc;
     % 
     % %--------------- Collecting results 
     % 
     % methods(r,1) = 'Nelder Mead' ;% 1st column method name
     % elapsed_times(r) = elapsed_time ; % second column elapsed time
     % grad_norm_last(r) = "//" ; % 3rd column grad_norm last
     % fk_last(r) = Fmin(end) ; % 4th column function value last
     % k_iterations(r) = k ; % 5th column k iteration
     % r = r+1;

    %----------------------------------------------------------------------
     %--------------------- Modified Newton Method Finn Diff
     % h = 8;
     % fd_gradf=@(x) findiff_grad(f,x, h, "fw");
     % fd_hessf=@(x) findiff_hess(f,x,h);
     % 
     % 
     % tic;
     % 
     % [xk, fk, gradfk_norm, k, xseq, btseq] = ...
     %       modified_newton_method(x0, f, fd_gradf, fd_hessf, kmax, ...
     %            tolgrad, c, rho, btmax);
     % 
     % elapsed_time = toc;
     % %--------------- Collecting results 
     % 
     % methods(r,1) = 'Newton Method MF Fin Diff' ;% 1st column method name
     % elapsed_times(r) = elapsed_time ; % second column elapsed time
     % fk_last(r) = fk(end) ; % 4th column function value last
     % k_iterations(r) = k ; % 5th column k iteration
     % r = r+1;
    
     %---------------------------------------------------------------------
     %--------------------- Truncated Newton Method MF Finn Diff
     
     % tic;
     % h = 8;
     % fd_gradf=@(x) findiff_grad(f,x, h, "fw");
     % fd_hessf=@(x) findiff_hess(f,x,h);
     % 
     % [xk, fk, gradfk_norm, k, xseq, btseq] = ...
     %       truncated_newton_method(x0, f, fd_gradf, fd_hessf, kmax, ...
     %            tolgrad, c, rho, btmax, pcg_maxit);
     % elapsed_time = toc;
     % 
     % %--------------- Collecting results 
     % 
     % methods(r,1) = 'Truncated Newton Method MF Fin Diff' ;% 1st column method name
     % elapsed_times(r) = elapsed_time ; % second column elapsed time
     % grad_norm_last(r) = gradfk_norm(end) ; % 3rd column grad_norm last
     % fk_last(r) = fk(end) ; % 4th column function value last
     % k_iterations(r) = k ; % 5th column k iteration


     %---------------------------------------------------------------------
     % ------- Printing Results 
 
     disp(['Results with N =  ', num2str(n) , ' Dimensions ']);
     disp(['% ---------------------------------','Results with N =  ', num2str(n) , ' Dimensions' ,' ---------------------%'])
     disp(['       Method          ','     k     ','elapsed_times   ', 'grad_norm_last    ' , 'fk_last      ' ]),
     disp([methods ,  fix(k_iterations), double(elapsed_times),double(grad_norm_last) , double(fk_last)]) ;
    
end  

