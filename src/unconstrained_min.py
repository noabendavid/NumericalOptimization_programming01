# 1. Line search minimization

import numpy as np

def calc_xk(x,direction,alpha):
    return x + alpha * direction

def wolfe_condition_backtracking(f, xk, pk, calc_hessian, c1=0.01, beta=0.5):   
    alpha = 1
    value, grad, _ = f(xk, calc_hessian)
    xk_new = calc_xk(xk,pk,alpha)
    value_new, gradient_new, _ = f(xk_new, calc_hessian)
    
    while value_new > value + c1 * alpha * np.dot(grad, pk): 
        alpha *= beta
        value_new, gradient_new, _ = f(calc_xk(xk,pk,alpha), calc_hessian)
    return alpha 


def line_search_minimization(f, x0=None, obj_tol=10**-8, param_tol=10**-12, max_iter=100, method=None, add_full_prints=True):
    
    if x0 is None:
        raise ValueError("Initial point x0 must be provided")
    if method not in ['gradient_descent', 'Newton']:
        raise ValueError("Method must be 'gradient_descent' or 'newton'")

    if add_full_prints:
        if method == 'Newton':
            print('#'*11 + '\n # Newton # \n' + '#'*11)
        else: 
            print('#'*21 + '\n # Gradient Descent # \n' + '#'*21)
           
    path = {}
    xk = x0
    calc_hessian = True if method == 'Newton' else False
    value, grad, hess = f(xk, calc_hessian)
    path[0] = {'location':xk, 'obj_value':value}

    for i in range(max_iter):
        
        if add_full_prints:
            print(f"Iteration {i}: xk = {xk}, f(xk) = {value}")

        if method == 'Newton':
            if hess is None:
                raise ValueError("Hessian function must be provided for Newton's method")
            try:
                pk = np.linalg.solve(hess, -grad)
            except:
                print('Hessian is singular or not square - break')
                print('*'*30)
                break
        else: 
            pk = -grad        
        
        alpha = wolfe_condition_backtracking(f, xk, pk, calc_hessian)

        xk_new = calc_xk(xk,pk,alpha)
        value_new, gradient_new, hessian_new = f(xk_new, calc_hessian)

        #Save path
        path[i+1] = {'location':xk_new, 'obj_value':value_new} 

        # Stop condition: reached min or converged
        if abs(value - value_new) < obj_tol or np.linalg.norm(xk_new - xk) < param_tol or not gradient_new.any():
            if add_full_prints:
                print(f"Reached stop condition at iteration {i+1}: xk = {xk_new}, f(xk) = {value_new}")
                print('*'*100 + '\n')
            return xk_new, value_new, True, path
        
        xk, value, grad, hess  = xk_new, value_new, gradient_new, hessian_new
    
    if add_full_prints:
                print(f"Minimum *not* found, finished at maximum iteration ({max_iter}): xk = {xk}, f(xk) = {value}")
                print('*'*100 + '\n')
    
    return xk, value, False, path
