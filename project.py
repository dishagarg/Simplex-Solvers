"""
Program to take matrices or vectors from csv and output an LP.
"""
import argparse
import csv
import numpy as np
import sys
from numpy import linalg


class Question2_2:
    """Question1."""

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
        min_b = float("inf")
        for row in data_b:
            self.b.append(row)
            if float(min(row)) < min_b:
                min_b = float(min(row))
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

        # Check for Infeasibility
        if min(z_star_n) < 0.0 and min_b < 0.0:
            print "Infeasible data"
            z_star_n = np.absolute(z_star_n)
            print z_star_n, matrix_B, matrix_N
            z_star_n, x_starB, matrix_B, matrix_N, self.Beta = self.dual_simplex_solver(x_starB, matrix_B, matrix_N, z_star_n)
            A_matrix = -1 * np.dot(linalg.inv(matrix_B), matrix_N)
            print np.array(self.Nu)
            sums = 0
            arr = []
            index_arr = []
            for i in range(len(self.c)):
                if i + 1 in self.Beta:
                    index = np.where(self.Beta == i + 1)
                    # print i, index
                    sums += self.c[i] * x_starB[index].squeeze()
                    arr.append(self.c[i] * (A_matrix[index, :].squeeze()))
                    index_arr.append(self.Nu)
            # arr = [[ 0.,  4.], [ 2.,  6.], [ 1.,  1.]]
            # index_arr = np.array([[ 0.,  4.], [ 0.,  4.], [ 0.,  4.]])
            index_arr = np.asarray(index_arr)
            print "self.Nu: ", index_arr
            arr_indx = np.asarray(arr)
            print self.c, arr_indx
            print "sums: ", sums
            another_arr = []
            for i in index_arr[0, :]:
                sumo = 0
                item = np.where(index_arr == i)
                sumo += np.sum(arr_indx[item].squeeze())
                print "jh: ", item, arr_indx[item], sumo, arr_indx[item].size
                if arr_indx[item].size != 0:
                    another_arr.append(sumo)

            another_arr = np.asarray(another_arr)
            print "hello", another_arr, sums, self.c, self.Nu
            for i in range(1, len(self.Nu) + 1):
                item = np.where(self.Nu == i)
                print "sume: ", item
                print another_arr, self.c
                sume = another_arr[item].squeeze() + self.c[item].squeeze()
                if sume.size != 0:
                    print "whe ", item
                    z_star_n[i-1] = another_arr[item].squeeze()+self.c[item].squeeze()
                else:
                    z_star_n[i-1] = another_arr[i-1].squeeze()
            if np.min(z_star_n) >= 0.0:
                print "Optimal Solution: ", sums
            else:
                self.c = 1*z_star_n
                print z_star_n, x_starB, self.c
                self.primal_simplex_solver(z_star_n, matrix_B, matrix_N, x_starB)

    def primal_simplex_solver(self, z_starN, matrix_B, matrix_N, x_star_b):
        """Linear Problem solver."""
        # step 1
        while np.min(z_starN) < 0.0:
            itemindex_j = np.where(z_starN == np.min(z_starN))
            j = self.Nu[itemindex_j[0]].squeeze()   # step 2 done

            # Step 3
            # Initialize e_j and e_i
            e_j = np.zeros(z_starN.shape)
            e_i = np.zeros(self.b.shape)
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
            i = self.Beta[itemindex[0]].squeeze()   # step 5 done

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

            self.Beta[itemindex[0]] = j
            self.Nu[itemindex_j[0]] = i
            matrix_N[:, itemindex_j[0]], matrix_B[:, itemindex[0]] = matrix_B[:, itemindex[0]], matrix_N[:, itemindex_j[0]]

        print "Optimal Solution found: "
        sol = 0
        for i in range(len(self.c)):
            if i + 1 in self.Beta:
                index = np.where(self.Beta == i + 1)
                sol += self.c[i].squeeze() * x_star_b[index].squeeze()
            else:
                sol += self.c[i] * 0
        print "Optimal Solution: ", sol
        print "x_star_b, z_starN: ", x_star_b, z_starN
        print "matrix_B, matrix_N: ", matrix_B, matrix_N
        print "self.Beta, self.Nu: ", self.Beta, self.Nu

    def dual_simplex_solver(self, x_starB, matrix_B, matrix_N, z_star_n):
        """Linear Problem solver."""
        # step 1
        while np.min(x_starB) < 0.0:
            itemindex_i = np.where(x_starB == np.min(x_starB))
            i = self.Beta[itemindex_i[0]].squeeze()   # step 2 done

            # Step 3
            # Initialize e_i and e_j
            e_i = np.zeros(x_starB.shape)
            e_j = np.zeros(self.c.shape)
            e_i[itemindex_i[0]] = 1
            mult = -1 * (np.transpose(np.dot(linalg.inv(matrix_B), matrix_N)))
            delta_z_n = (np.dot(mult, e_i))
            arr = np.array(delta_z_n / z_star_n)
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
            # Choose the smaller subscript
            if len(itemindex[0]) > 1:
                itemindex = tuple(np.asarray([[itemindex[0][0]], [itemindex[1][0]]]))
            j = self.Nu[itemindex[0]].squeeze()   # step 5 done

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

            self.Nu[itemindex[0]] = i
            self.Beta[itemindex_i[0]] = j
            matrix_N[:, itemindex[0]], matrix_B[:, itemindex_i[0]] = matrix_B[:, itemindex_i[0]], matrix_N[:, itemindex[0]]

        print "Optimal Solution found: "
        sol = 0
        for i in range(len(self.c)):
            if i + 1 in self.Beta:
                index = np.where(self.Beta == i + 1)
                sol += self.c[i].squeeze() * x_starB[index].squeeze()
            else:
                sol += self.c[i] * 0
        print "Optimal Solution: ", sol
        print "z_star_n, x_starB: ", z_star_n, x_starB
        print "matrix_B, matrix_N: ", matrix_B, matrix_N
        print "self.Beta, self.Nu: ", self.Beta, self.Nu
        return z_star_n, x_starB, matrix_B, matrix_N, self.Beta


if __name__ == "__main__":
    Question2_2()
