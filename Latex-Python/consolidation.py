"""
Program to take matrices or vectors from csv and output an LP.
"""
import argparse
import csv
import numpy as np
import sys
import project_part1 as pt1
import project_part2 as pt2
import project_part3 as pt3


class Project:
    """Question1 and 2."""

    def __init__(self, parent=None):
        """The start point."""
        parser = argparse.ArgumentParser(description='Linear Program Solver.')
        parser.add_argument('A_csv', help='The csv for A matrix.')
        parser.add_argument('b_csv', help='The csv for b vector.')
        parser.add_argument('c_csv', help='The csv for c vector.')
        args = parser.parse_args()

        # Fetch the data for A matrix
        data_a = csv.reader(open(args.A_csv, 'rb'))
        a = []
        for row in data_a:
            a.append(row)
        rows_a = len(a)
        cols_a = len(row)
        a = np.array([a]).reshape(rows_a, cols_a).astype(np.float)

        # Fetch the data for b vector
        data_b = csv.reader(open(args.b_csv, 'rb'))
        self.b = []
        for row in data_b:
            self.b.append(row)
        rows_b = len(self.b)
        cols_b = len(row)
        self.b = np.array([self.b]).reshape(rows_b, cols_b).astype(np.float)

        # Fetch the data for c vector
        data_c = csv.reader(open(args.c_csv, 'rb'))
        self.c = []
        for row in data_c:
            self.c.append(row)
        rows_c = len(self.c)
        cols_c = len(row)
        self.c = np.array([self.c]).reshape(rows_c, cols_c).astype(np.float)

        self.Nu = np.arange(1, rows_c + 1)
        self.Beta = np.arange(len(self.Nu) + 1, rows_b + len(self.Nu) + 1)

        split_a = np.hsplit(a, [rows_c, rows_b + rows_c])
        matrix_N = split_a[0]
        matrix_B = split_a[1]
        rows_N, cols_N = np.shape(matrix_N)
        rows_B, cols_B = np.shape(matrix_B)

        # Check for matrices consistency
        if cols_N == rows_c and rows_a == rows_b == rows_B == cols_B == rows_N:
            print("Everything is consistent!")
        else:
            print("[Error]: Data is inconsistent! Check the CSVs.")
            sys.exit()

        x_starB = self.b
        z_star_n = -1 * self.c
        objective = 0

        # For Primal Infeasibility
        if min(x_starB) < 0.0 and min(z_star_n) >= 0.0:
            print "[Error]: The problem is Primal Infeasible."
            print "So, performing Dual Simplex Method..."
            pt2.dual_simplex_solver(x_starB, matrix_B, matrix_N, z_star_n, self.b, self.c, self.Beta, self.Nu, objective)
            sys.exit()

        # For Dual Infeasibility
        if min(z_star_n) < 0.0 and min(x_starB) >= 0.0:
            print "[Error]: The problem is Dual Infeasible."
            print "So, performing Primal Simplex Method..."
            pt1.primal_simplex_solver(z_star_n, matrix_B, matrix_N, x_starB, self.b, self.c, self.Beta, self.Nu, objective)
            sys.exit()

        # For Dual and Primal Infeasibility
        if min(z_star_n) < 0.0 and min(x_starB) < 0.0:
            print "The problem is Dual and Primal Infeasible."
            pt3.primal_dual_simplex_solver(z_star_n, matrix_B, matrix_N, x_starB, self.b, self.c, self.Beta, self.Nu, objective)
            sys.exit()


if __name__ == "__main__":
    Project()
