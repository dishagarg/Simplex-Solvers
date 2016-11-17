"""
Program to take matrices or vectors from csv and output an LP.
"""
import numpy as np
from numpy import linalg
import project_part1 as pt1
import project_part2 as pt2


def primal_dual_simplex_solver(z_starN, matrix_B, matrix_N, x_star_b, b, c, Beta, Nu, optimal_value):
    """Primal Problem solver."""
    # step 1
    z_starN = np.absolute(z_starN)
    z_starN, x_star_b, matrix_B, matrix_N, Beta, sol = pt2.dual_simplex_solver(x_star_b, matrix_B, matrix_N, z_starN, b, c, Beta, Nu)
    A_matrix = -1 * np.dot(linalg.inv(matrix_B), matrix_N)
    # print "A_matrix: ", A_matrix
    # optimal_value = 0
    arr = []
    index_arr = []
    for i in range(len(c)):
        if i + 1 in Beta:
            index = np.where(Beta == i + 1)
            optimal_value += c[i] * x_star_b[index].squeeze()
            arr.append(c[i] * (A_matrix[index, :].squeeze()))
            index_arr.append(Nu)
    index_arr = np.asarray(index_arr)
    arr_indx = np.asarray(arr)
    another_arr = []
    for i in index_arr[0, :]:
        sumo = 0
        item = np.where(index_arr == i)
        sumo += np.sum(arr_indx[item].squeeze())
        if arr_indx[item].size != 0:
            another_arr.append(sumo)

    another_arr = np.asarray(another_arr)
    for i in range(1, len(Nu) + 1):
        item = np.where(Nu == i)
        sume = another_arr[item].squeeze() + c[item].squeeze()
        if sume.size != 0:
            z_starN[i - 1] = another_arr[item].squeeze() + c[item].squeeze()
        else:
            z_starN[i - 1] = another_arr[i - 1].squeeze()
    if np.min(z_starN) >= 0.0:
        print "Optimal Solution: ", optimal_value, z_starN
    else:
        c = 1 * z_starN
        pt1.primal_simplex_solver(z_starN, matrix_B, matrix_N, x_star_b, b, c, Beta, Nu)
    return z_starN, x_star_b, matrix_B, matrix_N, Beta, optimal_value
