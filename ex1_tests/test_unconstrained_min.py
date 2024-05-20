import unittest 
import matplotlib.pyplot as plt
import numpy as np
from ex1_tests.examples import quadratic_circles, quadratic_ellipses, quadratic_rotated_ellipses
from ex1_tests.examples import rosenbrock_func_example,linear_function, smoothed_corner_triangle_function
from src.unconstrained_min import line_search_minimization
from src.utils import plot_contour_lines, plot_function_values


def main_run(f, title, x0=[1.0,1.0], obj_tol=10**-8, param_tol=10**-12, max_iter=100, x_limits=None, y_limits=None, levels=100, add_full_prints=True, add_plot=True):
    
    print('*'*70 + f'\n***** Running test: {title} *****\n' + '*'*70)
    x0 = np.array(x0) if isinstance(x0, list) else x0

    # Call line search function with the relevant method
    gd_min, gd_value, gd_success, gd_path = line_search_minimization(f, x0, max_iter=max_iter, method='gradient_descent', add_full_prints=add_full_prints)
    if not add_full_prints:
        print(f"i) Gradient Descent Finished after {len(gd_path)-1} Iterrations: Final location at x = {gd_min}, with f(x) = {gd_value}, success = {gd_success}")

    nt_min, nt_value, nt_success, nt_path = line_search_minimization(f, x0, max_iter=max_iter, method='Newton', add_full_prints=add_full_prints)
    if not add_full_prints:
        print(f"ii) Newton Finished after {len(nt_path)-1} Iterrations: Final location at x = {nt_min}, with f(x) = {nt_value}, success = {nt_success}")

    if add_plot:
        all_paths_dict = {
                'Gradient Descent': [value['location'] for value in gd_path.values()],
                'Newton': [value['location'] for value in nt_path.values()]
            }

        func_values_dict = {
                'Gradient Descent': [value['obj_value'] for value in gd_path.values()],
                'Newton': [value['obj_value'] for value in nt_path.values()]
            }

        # Extracting the 'x' values from each array
        x_values = [array[0] for array in all_paths_dict['Gradient Descent'] + all_paths_dict['Newton']]

        # Add/ Remove epsilon to the graph bounderis limit
        e = 0.1
        
        # Finding the max and min 'x' values
        max_x = max(x_values) + e
        min_x = min(x_values) - e

        # Extracting the 'y' values from each array
        y_values = [array[1] for array in all_paths_dict['Gradient Descent'] + all_paths_dict['Newton']]

        # # Finding the max and min 'y' values
        max_y = max(y_values) + e
        min_y = min(y_values) - e

        # Create a figure with a grid of 1x2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1.5, 1]})

        # Create graph with the respective axes
        plot_contour_lines(f, (min_x, max_x), (min_y,max_y), paths=all_paths_dict, levels=100, fig=fig, ax=axs[0])
        plot_function_values(func_values_dict, ax=axs[1])

        # Add an overall title for the figure
        fig.suptitle(title, fontsize=16)

        # Show the plot
        plt.show()


class TestUnconstrainedMinimization(unittest.TestCase):
    
    def test_quadratic_example_1(self):
        main_run(quadratic_circles, 'Quadratic Example: Circles')

    def test_quadratic_example_2(self):
        main_run(quadratic_ellipses, 'Quadratic Example: Ellipses')
    
    def test_quadratic_example_3(self):
        main_run(quadratic_rotated_ellipses, 'Quadratic Example: Rotated Ellipses')
    
    def test_rosenbrock_func_example(self):
        main_run(rosenbrock_func_example, 'Rosenbrock Function', x0=[-1,2], max_iter=10000)

    def test_linear_function_example(self):
        main_run(linear_function, 'Linear Function')
    
    def test_exponential_function_example(self):
        main_run(smoothed_corner_triangle_function, 'Smoothed Corner Triangle Function')

if __name__ == '__main__':
    unittest.main()
