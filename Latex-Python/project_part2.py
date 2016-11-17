"""
Program to take matrices or vectors from csv and output an LP.
"""
import numpy as np
import sys
from numpy import linalg


def choose_smaller_subscript(itemindex):
    """Helper Function to choose the smaller subscript."""
    if len(itemindex[0]) > 1:
        itemindex = tuple(np.asarray([[itemindex[0][0]], [itemindex[1][0]]]))
    return itemindex


def dual_simplex_solver(x_starB, matrix_B, matrix_N, z_star_n, b, c, Beta, Nu, optimal_value):
    """Dual Problem solver."""
    # step 1
    while np.min(x_starB) < 0.0:
        itemindex_i = np.where(x_starB == np.min(x_starB))
        itemindex_i = choose_smaller_subscript(itemindex_i)
        i = Beta[itemindex_i[0]].squeeze()   # step 2 done

        # Step 3
        # Initialize e_i and e_j
        e_i = np.zeros(x_starB.shape)
        e_j = np.zeros(c.shape)
        e_i[itemindex_i[0]] = 1
        mult = -1 * (np.transpose(np.dot(linalg.inv(matrix_B), matrix_N)))
        delta_z_n = (np.dot(mult, e_i))
        import warnings

        def fxn():
            warnings.warn("deprecated", DeprecationWarning)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr = np.array(delta_z_n / z_star_n)
            fxn()
        # Step 4
        if any(np.isnan(ob) for ob in arr):
            index = np.where(np.isnan(arr))
            arr[index] = 0

        # Step 4
        s = np.reciprocal(np.max(arr).squeeze())
        if s < 0:
            print "[ERROR]: The dual is unbounded"
            sys.exit()
        itemindex = np.where(arr == np.max(arr))
        itemindex = choose_smaller_subscript(itemindex)
        j = Nu[itemindex[0]].squeeze()   # step 5 done

        # Step 6
        e_j[itemindex[0]] = 1
        mult = np.dot(linalg.inv(matrix_B), matrix_N)
        delta_x_Beta = np.dot(mult, e_j)

        # Step 7
        x_star_i = x_starB[itemindex_i[0]].squeeze()
        delta_x_i = delta_x_Beta[itemindex_i[0]].squeeze()
        t = x_star_i / delta_x_i

        # Step 8
        z_star_n = z_star_n - np.dot(s, delta_z_n)
        x_starB = x_starB - np.dot(t, delta_x_Beta)
        z_star_n[itemindex[0]] = s
        x_starB[itemindex_i[0]] = t

        Nu[itemindex[0]] = i
        Beta[itemindex_i[0]] = j
        matrix_N[:, itemindex[0]], matrix_B[:, itemindex_i[0]] = matrix_B[:, itemindex_i[0]], matrix_N[:, itemindex[0]]

    print "Optimal Solution found: "
    # Objective Function Comptation [c_B]'*[B^-1]*b
    # optimal_value = 0
    for i in range(len(c)):
        if i + 1 in Beta:
            index = np.where(Beta == i + 1)
            optimal_value += c[i].squeeze() * x_starB[index].squeeze()
        else:
            optimal_value += c[i] * 0
    print "Optimal Solution: ", optimal_value
    print "z_star_n, x_starB: ", z_star_n, x_starB
    print "matrix_B, matrix_N: ", matrix_B, matrix_N
    print "Beta, Nu: ", Beta, Nu
    return z_star_n, x_starB, matrix_B, matrix_N, Beta, optimal_value
