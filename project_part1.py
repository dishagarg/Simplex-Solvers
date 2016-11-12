"""
Program to take matrices or vectors from csv and output an LP.
"""
import numpy as np
import sys
from numpy import linalg


def primal_simplex_solver(z_starN, matrix_B, matrix_N, x_star_b, b, c, Beta, Nu):
    """Linear Problem solver."""
    # step 1
    while np.min(z_starN) < 0.0:
        itemindex_j = np.where(z_starN == np.min(z_starN))
        j = Nu[itemindex_j[0]].squeeze()   # step 2 done

        # Step 3
        # Initialize e_j and e_i
        e_j = np.zeros(z_starN.shape)
        e_i = np.zeros(b.shape)
        e_j[itemindex_j[0]] = 1
        mult = np.dot(linalg.inv(matrix_B), matrix_N)
        delta_x_b = np.dot(mult, e_j)
        arr = np.array(delta_x_b / x_star_b)
        # Step 4
        if any(np.isnan(ob) for ob in arr):
            index = np.where(np.isnan(arr))
            arr[index] = 0
        if np.max(arr).squeeze() <= 0:
            print "[Error]: The primal is Unbounded"
            sys.exit()
        t = np.reciprocal(np.max(arr).squeeze())
        itemindex = np.where(arr == np.max(arr))

        # Choose the smaller subscript
        if len(itemindex[0]) > 1:
            itemindex = tuple(np.asarray([[itemindex[0][0]], [itemindex[1][0]]]))
        i = Beta[itemindex[0]].squeeze()   # step 5 done

        # Step 6
        e_i[itemindex[0]] = 1
        mult = np.transpose(np.dot(linalg.inv(matrix_B), matrix_N))
        delta_z_Nu = -1 * (np.dot(mult, e_i))

        # Step 7
        z_star_j = z_starN[itemindex_j[0]].squeeze()
        delta_z_j = delta_z_Nu[itemindex_j[0]].squeeze()
        s = z_star_j / delta_z_j

        # Step 8
        x_star_b = x_star_b - np.dot(t, delta_x_b)
        z_starN = z_starN - np.dot(s, delta_z_Nu)
        x_star_b[itemindex[0]] = t
        z_starN[itemindex_j[0]] = s

        Beta[itemindex[0]] = j
        Nu[itemindex_j[0]] = i
        matrix_N[:, itemindex_j[0]], matrix_B[:, itemindex[0]] = matrix_B[:, itemindex[0]], matrix_N[:, itemindex_j[0]]

    print "Optimal Solution found: "
    sol = 0
    for i in range(len(c)):
        if i + 1 in Beta:
            index = np.where(Beta == i + 1)
            sol += c[i].squeeze() * x_star_b[index].squeeze()
        else:
            sol += c[i] * 0
    print "Optimal Solution: ", sol
    print "x_star_b, z_starN: ", x_star_b, z_starN
    print "matrix_B, matrix_N: ", matrix_B, matrix_N
    print "Beta, Nu: ", Beta, Nu
