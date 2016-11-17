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


def dual_simplex_solver(x_star_b, matrix_B, matrix_N, z_starN, b, c, Beta, Nu, optimal_value):
    """Dual Problem solver."""
    # Step 1: compute the optimal solution till xB < 0, if xB >= 0 then stop
    while np.min(x_star_b) < 0.0:
        # Step 2: Pick an index i in Beta for which min(x*B) < 0 (entering variable).
        itemindex_i = np.where(x_star_b == np.min(x_star_b))
        itemindex_i = choose_smaller_subscript(itemindex_i)
        i = Beta[itemindex_i[0]].squeeze()

        # Step 3: Compute Dual Step Direction delta_z_n
        # Initialize e_i and e_j
        e_i = np.zeros(x_star_b.shape)
        e_j = np.zeros(c.shape)
        e_i[itemindex_i[0]] = 1
        mult = -1 * (np.transpose(np.dot(linalg.inv(matrix_B), matrix_N)))
        delta_z_n = (np.dot(mult, e_i))

        # Suppress any divide by zero warnings
        import warnings

        def fxn():
            warnings.warn("deprecated", DeprecationWarning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            temp = np.array(delta_z_n / z_starN)
            fxn()
        # Step 4: Pick the largest s >= 0 for which every component of z*N remains nonnegative (Dual Step Length).
        if any(np.isnan(ob) for ob in temp):
            index = np.where(np.isnan(temp))
            temp[index] = 0

        # Step 5: The leaving variable is chosen with the max ratio.
        s = np.reciprocal(np.max(temp).squeeze())
        if s < 0:
            print "[ERROR]: The dual is unbounded"
            sys.exit()
        itemindex = np.where(temp == np.max(temp))
        itemindex = choose_smaller_subscript(itemindex)
        j = Nu[itemindex[0]].squeeze()   # step 5 done

        # Step 6: Compute Primal Step Direction delta_zN.
        e_j[itemindex[0]] = 1
        mult = np.dot(linalg.inv(matrix_B), matrix_N)
        delta_x_Beta = np.dot(mult, e_j)

        # Step 7: Compute Primal Step Length.
        x_star_i = x_star_b[itemindex_i[0]].squeeze()
        delta_x_i = delta_x_Beta[itemindex_i[0]].squeeze()
        t = x_star_i / delta_x_i

        # Step 8: Update Current Primal and Dual Solutions.
        # Check Degeneracy: if the ratio is infinite then no updation of x*B.
        if any(ob == float('inf') for ob in temp):
            z_starN = z_starN
        else:
            z_starN = z_starN - np.dot(s, delta_z_n)
            z_starN[itemindex[0]] = s
        x_star_b = x_star_b - np.dot(t, delta_x_Beta)
        x_star_b[itemindex_i[0]] = t

        # Step 9: Update Basis.
        Nu[itemindex[0]] = i
        Beta[itemindex_i[0]] = j
        matrix_N[:, itemindex[0]], matrix_B[:, itemindex_i[0]] = matrix_B[:, itemindex_i[0]], matrix_N[:, itemindex[0]]

    # Objective Function Comptation [c_B]'*[B^-1]*b
    print "\t[Optimal Solution found]"
    for i in range(len(c)):
        if i + 1 in Beta:
            index = np.where(Beta == i + 1)
            optimal_value += c[i].squeeze() * x_star_b[index].squeeze()
        else:
            optimal_value += c[i] * 0
    print "Optimal Solution: ", optimal_value
    print "x*B: \n", x_star_b
    print "z*N: \n", z_starN
    print "B: \n", matrix_B
    print "N: \n", matrix_N
    print "Beta: ", Beta
    print "Nu: ", Nu
    return z_starN, x_star_b, matrix_B, matrix_N, Nu, Beta, optimal_value
