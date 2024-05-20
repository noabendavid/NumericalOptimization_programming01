import numpy as np 


def quadratic_circles(x, hessian_flag):
    
    Q = np.array([[1, 0], [0, 1]])

    f, g = x.T.dot(Q).dot(x), 2 * Q.dot(x)
    # Hessian
    h = 2 * Q if hessian_flag else None
    
    return f, g, h

def quadratic_ellipses(x, hessian_flag):
    
    Q = np.array([[1, 0], [0, 100]])
    
    f, g = x.T.dot(Q).dot(x), 2 * Q.dot(x)
    
    h = 2 * Q if hessian_flag else None
    
    return f, g, h

def quadratic_rotated_ellipses(x, hessian_flag):
    
    Q1 = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
    Q2 = np.array([[100, 0], [0, 1]])
    Q = Q1.T.dot(Q2).dot(Q1)
    
    f, g = x.T.dot(Q).dot(x), 2 * Q.dot(x)

    h = 2 * Q if hessian_flag else None
    
    return f, g, h

def rosenbrock_func_example(x, hessian_flag):
    # Function value
    f = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2  

    # Gradient
    g = np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]), 
                  200 * (x[1] - x[0]**2)])  
    
    # Hessian
    h = None
    if hessian_flag:
        h = np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]],
                      [-400 * x[0], 200]])  
    
    return f, g, h

def linear_function(x, hessian_flag):
    a = np.array([1.0, 2.0])
    
    # Function value
    f = np.dot(a.T, x)
    
    # Gradient
    g = a

    # Hessian
    h = None
    if hessian_flag:
        h = np.zeros((2, 2))
    return f, g, h


def smoothed_corner_triangle_function(x, hessian_flag):
    # Function value
    f = np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1)
    
    # Gradient
    g = np.array([np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) - np.exp(-x[0] - 0.1),
                  3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)])
    
    # Hessian
    h = None
    if hessian_flag:
        h = np.array([[np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1),
                       3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)],
                      [3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1),
                       9 * np.exp(x[0] + 3 * x[1] - 0.1) + 9 * np.exp(x[0] - 3 * x[1] - 0.1)]])
    return f, g, h

