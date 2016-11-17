"""
Program to take matrices or vectors from csv and output an LP.
"""
import numpy as np
import sys
from numpy import linalg
import latexfile as tex

file_name = "Primal_Solver.tex"


def choose_smaller_subscript(itemindex):
    """Helper Function to choose the smaller subscript."""
    if len(itemindex[0]) > 1:
        itemindex = tuple(np.asarray([[itemindex[0][0]], [itemindex[1][0]]]))
    return itemindex


def primal_simplex_solver(z_starN, matrix_B, matrix_N, x_star_b, b, c, Beta, Nu, optimal_value):
    """Primal Problem solver."""
    # Step 1: compute the optimal solution till zN < 0, if zN >= 0 then stop
    count_iteration = 0
    initial_A = np.dot(linalg.inv(matrix_B), matrix_N)
    tex_file = tex.initial(file_name, "primal", initial_A, Beta, Nu, matrix_B, matrix_N, x_star_b, z_starN)
    while np.min(z_starN) < 0.0:
        count_iteration += 1
        # Step 2: Pick an index j in Nu for which min(z*j) < 0 (entering variable).
        tex.step1_primal(tex_file, count_iteration)

        itemindex_j = np.where(z_starN == np.min(z_starN))
        itemindex_j = choose_smaller_subscript(itemindex_j)
        j = Nu[itemindex_j[0]].squeeze()   # step 2 done
        tex.step2_primal(tex_file, j, np.min(z_starN))

        # Step 3: Compute Primal Step Direction delta_x_b
        # Initialize e_j and e_i
        e_j = np.zeros(z_starN.shape)
        e_i = np.zeros(b.shape)
        e_j[itemindex_j[0]] = 1
        mult = np.dot(linalg.inv(matrix_B), matrix_N)
        delta_x_b = np.dot(mult, e_j)
        tex.step3_primal(tex_file, mult, e_j, delta_x_b)

        # Suppress any divide by zero warnings
        import warnings

        def fxn():
            warnings.warn("deprecated", DeprecationWarning)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            temp = np.array(delta_x_b / x_star_b)
            fxn()

        # Step 4: Pick the largest t >= 0 for which every component of x*B remains nonnegative (Primal Step Length).
        if any(np.isnan(ob) for ob in temp):
            index = np.where(np.isnan(temp))
            temp[index] = 0
        if np.max(temp).squeeze() <= 0:
            print "[Error]: The primal is Unbounded"
            sys.exit()
        t = np.reciprocal(np.max(temp).squeeze())

        # Step 5: The leaving variable is chosen with the max ratio.
        itemindex = np.where(temp == np.max(temp))
        tex.step4_primal(tex_file, delta_x_b, x_star_b, t)
        itemindex = choose_smaller_subscript(itemindex)
        i = Beta[itemindex[0]].squeeze()   # step 5 done
        tex.step5_primal(tex_file, itemindex[0].squeeze() + 1, i)

        # Step 6: Compute Dual Step Direction delta_zN.
        e_i[itemindex[0]] = 1
        mult = np.transpose(np.dot(linalg.inv(matrix_B), matrix_N))
        delta_z_Nu = -1 * (np.dot(mult, e_i))
        tex.step6_primal(tex_file, mult, e_i, delta_z_Nu)

        # Step 7: Compute Dual Step Length.
        z_star_j = z_starN[itemindex_j[0]].squeeze()
        delta_z_j = delta_z_Nu[itemindex_j[0]].squeeze()
        s = z_star_j / delta_z_j
        tex.step7_primal(tex_file, z_star_j, delta_z_j, s)

        # Step 8: Update Current Primal and Dual Solutions.
        # Check Degeneracy: if the ratio is infinite then no updation of x*B.
        if any(ob == float('inf') for ob in temp):
            x_star_b = x_star_b
        else:
            matrix_xstar_b = x_star_b - np.dot(t, delta_x_b)
        matrix_zstar_n = z_starN - np.dot(s, delta_z_Nu)
        tex.step8_primal(tex_file, t, s, j, t, x_star_b, delta_x_b, matrix_xstar_b, i, s, z_starN, delta_z_Nu, matrix_zstar_n)
        matrix_zstar_n[itemindex_j[0]] = s
        if any(ob == float('inf') for ob in temp):
            x_star_b = x_star_b
        else:
            matrix_xstar_b[itemindex[0]] = t

        # Step 9: Update Basis.
        Beta[itemindex[0]] = j
        Nu[itemindex_j[0]] = i
        matrix_N[:, itemindex_j[0]], matrix_B[:, itemindex[0]] = matrix_B[:, itemindex[0]], matrix_N[:, itemindex_j[0]]
        x_star_b = matrix_xstar_b
        z_starN = matrix_zstar_n
        tex.step9_primal(tex_file, Beta, Nu, matrix_B, matrix_N, x_star_b, z_starN)

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
    count_iteration += 1
    tex.step1_primal_over(tex_file, c, optimal_value, count_iteration)
    tex.end_document(tex_file, file_name)
    return z_starN, x_star_b, matrix_B, matrix_N, Nu, Beta, optimal_value
