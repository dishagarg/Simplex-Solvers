"""
Program to take matrices or vectors from csv and output an LP.
"""
import numpy as np
from numpy import linalg
import project_part1 as pt1
import project_part2 as pt2


def primal_dual_simplex_solver(z_starN, matrix_B, matrix_N, x_star_b, b, c, Beta, Nu, optimal_value):
    """Primal Problem solver."""
    # Step 1: Convert z*N to nonnegative to make it Dual Feasible
    temp_c = -1 * np.ones((len(c), 1))
    z_starN = -1 * temp_c
    z_starN, x_star_b, matrix_B, matrix_N, Nu, Beta, sol = pt2.dual_simplex_solver(x_star_b, matrix_B, matrix_N, z_starN, b, temp_c, Beta, Nu, optimal_value)

    A_matrix = -1 * np.dot(linalg.inv(matrix_B), matrix_N)

    row_matrix_a = []
    indices_rows_matrix_a = []
    # First Loop to pull x*B and corresponding A matrix row for the decision variables
    # and multiply them to their coefficients in the objective function.
    for i in range(len(c)):
        if i + 1 in Beta:
            index = np.where(Beta == i + 1)
            optimal_value += c[i] * x_star_b[index].squeeze()
            row_matrix_a.append(c[i] * (A_matrix[index, :].squeeze()))
            indices_rows_matrix_a.append(Nu)
    indices_rows_matrix_a = np.asarray(indices_rows_matrix_a)
    array_row_matrix_a = np.asarray(row_matrix_a)
    summed_array_to_substitute = []
    # Second Loop to check if we have multiple rows pulled out from matrix_a. Then check the
    # coefficients for the same decision variables and then add them.
    for i in indices_rows_matrix_a[0, :]:
        temp = 0
        item = np.where(indices_rows_matrix_a == i)
        temp += np.sum(array_row_matrix_a[item].squeeze())
        if array_row_matrix_a[item].size != 0:
            summed_array_to_substitute.append(temp)

    # Now we got the summed array for the rows we substituted in original objective
    # So, Third Loop is to check and sum if we have multiple coefficients for the same
    # decision variables in this new objective function.
    summed_array_to_substitute = np.asarray(summed_array_to_substitute)
    for i in range(1, len(Nu) + 1):
        item = np.where(Nu == i)
        temp_sum = summed_array_to_substitute[item].squeeze() + c[item].squeeze()
        if temp_sum.size != 0:
            temp_c[i - 1] = summed_array_to_substitute[item].squeeze() + c[item].squeeze()
        else:
            temp_c[i - 1] = summed_array_to_substitute[i - 1].squeeze()
    z_starN = -1 * temp_c
    if np.min(z_starN) >= 0.0:
        print "\t[Optimal Solution found]"
        print "Optimal Solution is: ", optimal_value
        print "z*N: \n", z_starN
    else:
        pt1.primal_simplex_solver(z_starN, matrix_B, matrix_N, x_star_b, b, c, Beta, Nu, optimal_value)
    return z_starN, x_star_b, matrix_B, matrix_N, Nu, Beta, optimal_value
