"""
Non-Basis and Basis functions for Criss-Cross method.
"""
import numpy as np
import sys
from numpy import linalg
import criss_cross as cs


def non_basis(entering_index, e_i, e_j, z_starN, matrix_B, matrix_N, x_star_b, Beta, Nu):
    j = Nu[entering_index[0]].squeeze()
    e_j[entering_index[0]] = 1
    # Step 4: Compute Primal Step Direction delta_x_b
    mult = np.dot(linalg.inv(matrix_B), matrix_N)
    delta_x_b = np.dot(mult, e_j)
    # Step 5: The leaving variable is chosen as that positive variable in delta_xb which has smallest subscript.
    column_indices = np.where(delta_x_b > 0.0)
    leaving_index = cs.choose_smaller_subscript(column_indices)
    # Suppress any divide by zero warnings
    import warnings

    def fxn():
        warnings.warn("deprecated", DeprecationWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temp = np.array(delta_x_b / x_star_b)
        fxn()

    if any(np.isnan(ob) for ob in temp):
        index = np.where(np.isnan(temp))
        temp[index] = 0
    if np.max(temp).squeeze() <= 0:
        print "[Error]: The primal is Unbounded"
        sys.exit()

    # Step 6: Pick t as Primal Step Length.
    t = np.reciprocal(temp[leaving_index[0]].squeeze())
    i = Beta[leaving_index[0]].squeeze()
    e_i[leaving_index[0]] = 1

    # Step 7: Compute Dual Step Direction delta_zNu.
    mult = np.transpose(np.dot(linalg.inv(matrix_B), matrix_N))
    delta_z_Nu = -1 * (np.dot(mult, e_i))

    # Step 8: Pick s as Dual Step Length.
    z_star_j = z_starN[entering_index[0]].squeeze()
    delta_z_j = delta_z_Nu[entering_index[0]].squeeze()
    s = z_star_j / delta_z_j

    # Step 9: Update Current Primal and Dual Solutions.
    # Check Degeneracy: if the ratio is infinite then no updation of x*B.
    if any(ob == float('inf') for ob in temp):
        x_star_b = x_star_b
    else:
        x_star_b = x_star_b - np.dot(t, delta_x_b)
        x_star_b[leaving_index[0]] = t
    z_starN = z_starN - np.dot(s, delta_z_Nu)
    z_starN[entering_index[0]] = s

    # Step 10: Update Basis.
    Beta[leaving_index[0]] = j
    Nu[entering_index[0]] = i
    matrix_N[:, entering_index[0]], matrix_B[:, leaving_index[0]] = matrix_B[:, leaving_index[0]], matrix_N[:, entering_index[0]]
    return z_starN, matrix_B, matrix_N, x_star_b, Beta, Nu


def basis(leaving_index, e_i, e_j, z_starN, matrix_B, matrix_N, x_star_b, Beta, Nu):
    i = Beta[leaving_index[0]].squeeze()
    e_i[leaving_index[0]] = 1
    # Step 4: Compute Primal Step Direction delta_x_b
    mult = np.transpose(np.dot(linalg.inv(matrix_B), matrix_N))
    delta_z_Nu = -1 * (np.dot(mult, e_i))
    # Step 5: The entering variable is chosen as that positive variable in delta_z_Nu which has smallest subscript.
    column_indices = np.where(delta_z_Nu > 0.0)
    entering_index = cs.choose_smaller_subscript(column_indices)
    # Suppress any divide by zero warnings
    import warnings

    def fxn():
        warnings.warn("deprecated", DeprecationWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temp = np.array(delta_z_Nu / z_starN)
        fxn()

    if any(np.isnan(ob) for ob in temp):
        index = np.where(np.isnan(temp))
        temp[index] = 0
    if np.max(temp).squeeze() <= 0:
        print "[Error]: The dual is Unbounded"
        sys.exit()

    # Step 6: Pick s as Dual Step Length.
    s = np.reciprocal(temp[entering_index[0]].squeeze())
    j = Nu[entering_index[0]].squeeze()
    e_j[entering_index[0]] = 1

    # Step 7: Compute Primal Step Direction delta_x_b.
    mult = np.dot(linalg.inv(matrix_B), matrix_N)
    delta_x_b = np.dot(mult, e_j)

    # Step 8: Pick s as Primal Step Length.
    x_star_i = x_star_b[leaving_index[0]].squeeze()
    delta_x_i = delta_x_b[leaving_index[0]].squeeze()
    t = x_star_i / delta_x_i

    # Step 9: Update Current Primal and Dual Solutions.
    # Check Degeneracy: if the ratio is infinite then no updation of x*B.
    if t == float('inf'):
        x_star_b = x_star_b
    else:
        x_star_b = x_star_b - np.dot(t, delta_x_b)
        x_star_b[leaving_index[0]] = t
    z_starN = z_starN - np.dot(s, delta_z_Nu)
    z_starN[entering_index[0]] = s

    # Step 10: Update Basis.
    Beta[leaving_index[0]] = j
    Nu[entering_index[0]] = i
    matrix_N[:, entering_index[0]], matrix_B[:, leaving_index[0]] = matrix_B[:, leaving_index[0]], matrix_N[:, entering_index[0]]
    return z_starN, matrix_B, matrix_N, x_star_b, Beta, Nu
