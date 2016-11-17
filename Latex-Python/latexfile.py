"""Code to bind python with LaTeX."""
import os
from itertools import izip

# Variables for LaTeX templates insertion
initial_template = '''\documentclass [12pt] {article}
\usepackage{amsmath}
\usepackage{url}
\usepackage[super]{nth}
\pagestyle{plain}
\\begin{document}
'''
initial_vector_description = '''The matrix A is given by
\[
%(initial_A)s.
\]
The initial sets of basic and nonbasic indices are
\[
{\mathcal B} = \left \{ %(Beta)s \\right \}  \quad
and    \quad
{\mathcal N} = \left \{ %(Nu)s \\right \}.
\]
Corresponding to these sets, we have the submatrices of A:
\[
B = %(initial_B)s   \quad
N = %(initial_N)s,
\]
and the initial values of the basic variables are given by
\[
 x^*_{\mathcal B} = b = %(initial_xstar_b)s ,   
\]
and the initial nonbasic dual variables are simply
\[
z^*_{\mathcal N} = -c_{\mathcal N} = %(initial_zstar_n)s.
\]
'''
method_primal = '''
\section{Primal Simplex Method.}
'''
method_dual = '''
\section{Dual Simplex Method.}
'''
method_primal_dual = '''
\section{Primal-Dual Simplex Method.}
'''
step1_primal_true = '''
\subsection{ \\nth{%(count_iteration)s} Iteration.}
\\textit{Step 1. } Since \\textit{z}$^*_{\mathcal N}$ has some negative components, the current solution is not optimal.\\\\
'''
step2_primal_true = '''\\textit{Step 2. } Since \\textit{z}$^*_%(entering_index)s$ = %(entering_value)s and this is the most negative of the two nonbasic dual variables, we see that the entering index is
\[
j = %(entering_index)s.
\]
'''
step3_primal_true = '''\\textit{Step 3. }
\[
\Delta X_{\mathcal B} = B^{-1} N e_j = %(matrix_BN)s
%(matrix_ej)s
= %(matrix_delta_xb)s
.
\]
'''
step4_primal_true = '''\\textit{Step 4. }
\[
t = \left ( max \left \{ %(matrix_fraction)s \\right \} \\right )^{-1}   = %(max_ratio)s.
\]'''
step5_primal_true = '''\\textit{Step 5.} Since the ratio that achieved the maximum in Step 4 was the \\nth{%(max_ratio_number)s} ratio and this ratio corresponds to basis index %(max_ratio_index)s, we see that
\[
i = %(max_ratio_index)s.
\]
'''
step6_primal_true = '''\\textit{Step 6. } 
\[
\Delta z_{\mathcal N} = -\left (B^{-1} N \\right )^T e_i = - %(matrix_BN_T)s
%(matrix_ei)s
= %(matrix_delta_zn)s.
\]
'''
step7_primal_true = '''\\textit{Step 7. } 
\[
s =  \\frac{z^*_j}{\Delta z_j} = %(fraction_zstar_delta)s.
\]
'''
step8_primal_true = '''\\textit{Step 8.}
\[
x^*_%(index_xstar)s = %(value_xstar)s,        x^*_{\mathcal B} = 
%(matrix_xstar)s - %(value_t)s %(matrix_delta_xb)s = %(matrix_xstar_b)s ,
\]
\[
z^*_%(index_zstar)s = %(value_zstar)s,        z^*_{\mathcal N} = 
%(matrix_zstar)s - %(value_s)s %(matrix_delta_zn)s = %(matrix_zstar_n)s .
\]
'''
step9_primal_true = '''\\textit{Step 9.} The new sets of basic and nonbasic indices are
\[
{\mathcal B} = \left \{ %(Beta)s \\right \}  \quad
and    \quad
{\mathcal N} = \left \{ %(Nu)s \\right \}.
\]
Corresponding to these sets, we have the new basic and nonbasic submatrices of A,
\[
B = %(matrix_B)s  \quad
N = %(matrix_N)s,
\]
and the new basic primal variables and nonbasic dual variables:
\[
 x^*_{\mathcal B} = 
%(matrix_xstar_indexedBeta)s = %(matrix_xstar_final)s ,   \quad
z^*_{\mathcal N} = 
%(matrix_zstar_indexedNu)s = %(matrix_zstar_final)s.
\]
'''
step1_primal_false = '''
\subsection{ \\nth{%(count_iteration)s} Iteration.}
\\textit{Step 1. } Since \\textit{z}$^*_{\mathcal N}$ has all nonnegative components, the current solution is optimal. The optimal objective function value is
\[
\zeta^* = %(objective_function)s = %(optimal_value)s
\]
'''


def display_objective_function(matrix_c):
    """Helper function to display the objective function in LaTeX."""
    temp = []
    list_c = matrix_c.squeeze().tolist()
    for i in range(len(list_c)):
        strn = str(list_c[i]) + "x^*_" + str(i + 1)
        temp.append(strn)
        temp.append(" + ")
    temp.pop(-1)
    string = ' '.join(str(i) for i in temp)
    return string


def display_matrix_with_indices(arrays, x_or_z):
    """Helper function to display a matrix having x* or z* with indices in LaTeX."""
    temp = []
    temp.append("\\begin{bmatrix}")
    for i in arrays:
        if x_or_z == "x":
            strn = "x^*_" + str(i)
        elif x_or_z == "z":
            strn = "z^*_" + str(i)
        temp.append(strn)
        temp.append('\\\\')
    temp.pop(-1)
    temp.append("\end{bmatrix}")
    string = ' '.join(str(i) for i in temp)
    return string


def display_matrix(matrix):
    """Helper function to display a matrix in LaTeX."""
    temp = []
    temp.append("\\begin{bmatrix}")
    for i in matrix:
        for j in range(len(i) - 1):
            temp.append(i[j])
            temp.append('&')
        temp.append(i[len(i) - 1])
        temp.append('\\\\')
    temp.append("\end{bmatrix}")
    string = ' '.join(str(i) for i in temp)
    return string


def display_fractions(matrix_a, matrix_b):
    """Helper function to display fractions between two matrices in LaTeX."""
    temp = []
    for i, j in izip(matrix_a, matrix_b):
        strn = "\\frac{" + str(float(i)) + "}{" + str(float(j)) + "}"
        temp.append(strn)
        temp.append(",")
    temp.pop(-1)
    string = ' '.join(str(i) for i in temp)
    return string.strip('()')


def initial(file_name, method, initial_A, Beta, Nu, initial_B, initial_N, initial_xstar_b, initial_zstar_n):
    """Initial function to begin the LaTeX document."""
    tex_file = file(file_name, "w+")
    tex_file.writelines(initial_template)
    if method == "primal":
        tex_file.writelines(method_primal)
    elif method == "dual":
        tex_file.writelines(method_dual)
    else:
        tex_file.writelines(method_primal_dual)
    initial_A = display_matrix(initial_A)
    initial_B = display_matrix(initial_B)
    initial_N = display_matrix(initial_N)
    Beta = str(Beta.tolist()).strip('[]')
    Nu = str(Nu.tolist()).strip('[]')
    initial_xstar_b = display_matrix(initial_xstar_b)
    initial_zstar_n = display_matrix(initial_zstar_n)
    string = (initial_vector_description % {'initial_A': initial_A, 'Beta': Beta, 'Nu': Nu, 'initial_B': initial_B, 'initial_N': initial_N, 'initial_xstar_b': initial_xstar_b,
                                            'initial_zstar_n': initial_zstar_n})
    tex_file.writelines(string)
    return tex_file


def step1_primal(tex_file, count_iteration):
    """Step 1 of Primal, if any element in z_star_n is negative."""
    string = (step1_primal_true % {'count_iteration': count_iteration})
    tex_file.writelines(string)


def step1_primal_over(tex_file, matrix_c, optimal_value, count_iteration):
    """Last Iteration of Primal, if all elements in z_star_n are nonnegative."""
    objective_function = display_objective_function(matrix_c)
    string = (step1_primal_false % {'objective_function': objective_function, 'optimal_value': optimal_value, 'count_iteration': count_iteration})
    tex_file.writelines(string)


def step2_primal(tex_file, entering_index, entering_value):
    """Step 2 of Primal, display the selection of entering variable in z_star_n."""
    string = (step2_primal_true % {'entering_index': entering_index, 'entering_value': entering_value})
    tex_file.writelines(string)


def step3_primal(tex_file, matrix_BN, matrix_ej, matrix_delta_xb):
    """Step 3 of Primal, display the computation of Primal Step Direction (matrix_delta_xb)."""
    matrix_BN = display_matrix(matrix_BN)
    matrix_ej = display_matrix(matrix_ej)
    matrix_delta_xb = display_matrix(matrix_delta_xb)
    string = (step3_primal_true % {'matrix_BN': matrix_BN, 'matrix_ej': matrix_ej, 'matrix_delta_xb': matrix_delta_xb})
    tex_file.writelines(string)


def step4_primal(tex_file, matrix_delta_xb, matrix_x_star_b, max_ratio):
    """Step 4 of Primal, display the computation of Primal Step Length (max_ratio)."""
    matrix_fraction = display_fractions(matrix_delta_xb, matrix_x_star_b)
    string = (step4_primal_true % {'matrix_fraction': matrix_fraction, 'max_ratio': max_ratio})
    tex_file.writelines(string)


def step5_primal(tex_file, max_ratio_number, max_ratio_index):
    """Step 5 of Primal, display the selection of Leaving Variable (max_ratio_number)."""
    string = (step5_primal_true % {'max_ratio_number': max_ratio_number, 'max_ratio_index': max_ratio_index})
    tex_file.writelines(string)


def step6_primal(tex_file, matrix_BN_T, matrix_ei, matrix_delta_zn):
    """Step 6 of Primal, display the computation of Dual Step Direction deltazN (matrix_delta_zn)."""
    matrix_BN_T = display_matrix(matrix_BN_T)
    matrix_ei = display_matrix(matrix_ei)
    matrix_delta_zn = display_matrix(matrix_delta_zn)
    string = (step6_primal_true % {'matrix_BN_T': matrix_BN_T, 'matrix_ei': matrix_ei, 'matrix_delta_zn': matrix_delta_zn})
    tex_file.writelines(string)


def step7_primal(tex_file, z_star_j, delta_z_j, s):
    """Step 7 of Primal, display the computation of Dual Step Length (s)."""
    fraction_zstar_delta = "\\frac{{ {} }}{{ {} }} = {}".format(z_star_j, delta_z_j, s)
    string = (step7_primal_true % {'fraction_zstar_delta': fraction_zstar_delta})
    tex_file.writelines(string)


def step8_primal(tex_file, value_t, value_s, index_xstar, value_xstar, x_star_b, delta_x_b, matrix_xstar_b, index_zstar, value_zstar, z_starN, delta_z_Nu, matrix_zstar_n):
    """Step 8 of Primal, display the updation of Current Primal and Dual Solutions."""
    matrix_xstar = display_matrix(x_star_b)
    matrix_delta_xb = display_matrix(delta_x_b)
    matrix_xstar_b = display_matrix(matrix_xstar_b)
    matrix_zstar = display_matrix(z_starN)
    matrix_delta_zn = display_matrix(delta_z_Nu)
    matrix_zstar_n = display_matrix(matrix_zstar_n)
    string = (step8_primal_true % {'value_t': value_t, 'value_s': value_s, 'index_xstar': index_xstar, 'value_xstar': value_xstar, 'matrix_xstar': matrix_xstar,
                                   'matrix_delta_xb': matrix_delta_xb, 'matrix_xstar_b': matrix_xstar_b, 'index_zstar': index_zstar, 'value_zstar': value_zstar,
                                   'matrix_zstar': matrix_zstar, 'matrix_delta_zn': matrix_delta_zn, 'matrix_zstar_n': matrix_zstar_n})
    tex_file.writelines(string)


def step9_primal(tex_file, Beta, Nu, matrix_B, matrix_N, x_star_b, z_starN):
    """Step 9 of Primal, display the updation of Basis."""
    matrix_xstar_indexedBeta = display_matrix_with_indices(Beta, "x")
    matrix_zstar_indexedNu = display_matrix_with_indices(Nu, "z")
    Beta = str(Beta.tolist()).strip('[]')
    Nu = str(Nu.tolist()).strip('[]')
    matrix_B = display_matrix(matrix_B)
    matrix_N = display_matrix(matrix_N)
    matrix_xstar_final = display_matrix(x_star_b)
    matrix_zstar_final = display_matrix(z_starN)
    string = (step9_primal_true % {'matrix_B': matrix_B, 'matrix_N': matrix_N, 'matrix_xstar_final': matrix_xstar_final, 'matrix_zstar_final': matrix_zstar_final,
                                   'matrix_xstar_indexedBeta': matrix_xstar_indexedBeta, 'matrix_zstar_indexedNu': matrix_zstar_indexedNu, 'Beta': Beta, 'Nu': Nu})
    tex_file.writelines(string)


def end_document(tex_file, file_name):
    """Last Function to end the document and execute the tex file."""
    last_line = "\end{document}"
    tex_file.writelines(last_line)
    tex_file.close()

	# Command to create a pdf
    os.system("pdflatex {}".format(file_name))

    # Cleaning unnecessary files
    os.system('rm *.dvi *.ps')
    if os.path.isfile(file_name.replace('.tex', '.log')):
        os.system('rm *.log')
    if os.path.isfile(file_name.replace('.tex', '.aux')):
        os.system('rm *.aux')
    if os.path.isfile(file_name.replace('.tex', '.bbl')):
        os.system('rm *.bbl')
    if os.path.isfile(file_name.replace('.tex', '.blg')):
        os.system('rm *.blg')
