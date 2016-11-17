"""
Program to take matrices or vectors from csv and output an LP.
"""
import numpy as np
import basis_nonbasis as nb


def choose_smaller_subscript(itemindex):
    """Helper Function to choose the smaller subscript."""
    if len(itemindex[0]) > 1:
        itemindex = tuple(np.asarray([[itemindex[0][0]], [itemindex[1][0]]]))
    return itemindex


def criss_cross_solver(z_starN, matrix_B, matrix_N, x_star_b, b, c, Beta, Nu, optimal_value):
    """Criss Cross solver."""
    # Step 1: compute the optimal solution till zN < 0 and xB < 0, stop if both are positive.
    while np.min(z_starN) < 0.0 or np.min(x_star_b) < 0.0:
        # Step 2: Pick indices in z_starN, x_star_b where the coefficients are negative.
        nonbasis_indices = np.where(z_starN < 0.0)
        basis_indices = np.where(x_star_b < 0.0)
        # Step 3: Pick the index which has smaller subscript for the decision variables (entering variable).
        nonbasic_entering_index = choose_smaller_subscript(nonbasis_indices)
        basic_leaving_index = choose_smaller_subscript(basis_indices)

        # Initialize e_j and e_i
        e_j = np.zeros(z_starN.shape)
        e_i = np.zeros(b.shape)

        # if there is a non-basic infeasible variable
        if len(nonbasic_entering_index[0]) > 0:
            z_starN, matrix_B, matrix_N, x_star_b, Beta, Nu = nb.non_basis(nonbasic_entering_index, e_i, e_j, z_starN, matrix_B, matrix_N, x_star_b, Beta, Nu)
        else:
            z_starN, matrix_B, matrix_N, x_star_b, Beta, Nu = nb.basis(basic_leaving_index, e_i, e_j, z_starN, matrix_B, matrix_N, x_star_b, Beta, Nu)

    # Objective Function Comptation [c_B]'*[B^-1]*b
    optimal_value = 0
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
