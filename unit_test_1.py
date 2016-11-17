import unittest
import numpy as np
import project_part1 as pt1
from numpy import linalg


class UnitTest1(unittest.TestCase):
    """First test case."""

    # preparing to test
    def setUp(self):
        """Setting up for the test."""
		# Primal feasible problem
        self.a = np.asarray([[1, 4, 0, 1, 0, 0], [3, -1, 1, 0, 1, 0]]).astype(np.float)
        self.b = np.asarray([[1], [3]]).astype(np.float)
        self.c = np.asarray([[4], [1], [3]]).astype(np.float)
		
		# Optimal Solution to verify
        self.optimal_solution = 10.0
        self.x_b_solution = np.asarray([[0.25], [3.25]]).astype(np.float)
        self.z_n_solution = np.asarray([[1], [6], [3]]).astype(np.float)
        self.b_solution = np.asarray([[4, 0], [-1, 1]]).astype(np.float)
        self.n_solution = np.asarray([[1, 1, 0], [0, 3, 1]]).astype(np.float)
        self.beta_solution = [2, 3]
        self.nu_solution = [4, 1, 5]

        rows_b, cols_b = np.shape(self.b)
        rows_c, cols_c = np.shape(self.c)
        self.Nu = np.arange(1, rows_c + 1)
        self.Beta = np.arange(len(self.Nu) + 1, rows_b + len(self.Nu) + 1)

        split_a = np.hsplit(self.a, [rows_c, rows_b + rows_c])
        self.matrix_N = split_a[0]
        self.matrix_B = split_a[1]
        self.x_starB = self.b
        self.z_star_n = -1 * self.c
        self.objective = 0

    def test_1_consistency(self):
        """Check for matrices consistency."""
        rows_a, cols_a = np.shape(self.a)
        rows_b, cols_b = np.shape(self.b)
        rows_c, cols_c = np.shape(self.c)

        rows_N, cols_N = np.shape(self.matrix_N)
        rows_B, cols_B = np.shape(self.matrix_B)
        self.assertEqual(cols_N, rows_c)
        self.assertEqual(rows_a, rows_b)
        self.assertEqual(rows_b, rows_B)
        self.assertEqual(rows_B, cols_B)
        self.assertEqual(rows_N, rows_B)
        print("Everything is consistent!")

    def test_2_primal_feasibility(self):
        """Check for feasibility."""
        self.assertLess(min(self.z_star_n), 0.0, "Sorry, the problem is not Dual Infeasible.")
        self.assertGreaterEqual(min(self.x_starB), 0.0, "Sorry, the problem is not Primal Feasible.")

    def test_3_primal_solution(self):
        """Check for Primal Optimal Solution."""
        print "Solving by Primal Simplex."
        z_starN1, x_star_b1, matrix_B1, matrix_N1, Nu1, Beta1, opt_value1 = pt1.primal_simplex_solver(self.z_star_n, self.matrix_B,
                                                                                                      self.matrix_N, self.x_starB, self.b,
                                                                                                      self.c, self.Beta, self.Nu, self.objective)
        xb = np.dot(linalg.inv(matrix_B1), self.b)
        self.assertEqual(xb.all(), x_star_b1.all(), msg="x*b != inv(B) * b")
        self.assertGreaterEqual(z_starN1.all(), 0.00000)
        self.assertAlmostEqual(self.optimal_solution, opt_value1, 3)
        self.assertAlmostEqual(self.x_b_solution.all(), x_star_b1.all(), 3)
        self.assertAlmostEqual(self.z_n_solution.all(), z_starN1.all(), 3)
        self.assertAlmostEqual(self.b_solution.all(), matrix_B1.all(), 3)
        self.assertAlmostEqual(self.n_solution.all(), matrix_N1.all(), 3)
        self.assertEqual(all(self.beta_solution), all(Beta1))
        self.assertEqual(all(self.nu_solution), all(Nu1))

    # ending the test
    def tearDown(self):
        """Cleaning up after the test."""
        pass


if __name__ == '__main__':
    unittest.main()
