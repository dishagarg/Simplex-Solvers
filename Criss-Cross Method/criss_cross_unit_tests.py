import unittest
import numpy as np
import project_part3 as pt3
import criss_cross as cs


class UnitTest1(unittest.TestCase):
    """First test case."""

    # preparing to test
    def setUp(self):
        """Setting up for the test."""
		# Infeasible problem
        self.a = np.asarray([[2, 1, 1, 0], [-2, 2, 0, 1]]).astype(np.float)
        self.b = np.asarray([[4], [-2]]).astype(np.float)
        self.c = np.asarray([[1], [1]]).astype(np.float)
		
		# Optimal Solution to verify
        self.optimal_solution = 2.3333
        self.x_b_solution = np.asarray([[0.6666], [1.6666]]).astype(np.float)
        self.z_n_solution = np.asarray([[0.1666], [0.6666]]).astype(np.float)
        self.b_solution = np.asarray([[1, 2], [2, -2]]).astype(np.float)
        self.n_solution = np.asarray([[0, 1], [1, 0]]).astype(np.float)
        self.beta_solution = [2, 1]
        self.nu_solution = [4, 3]

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

    def test_2_dual_primal_infeasible(self):
        """Check if the problem is both dual and primal infeasible."""
        self.assertLess(min(self.z_star_n), 0.0)
        self.assertLess(min(self.x_starB), 0.0)
        print "The problem is Dual and Primal Infeasible."

    def test_3_optimal_solution_simplex(self):
        """Check optimal solution from Simplex Method to compare with CrissCross."""
        z_starN1, x_star_b1, matrix_B1, matrix_N1, Nu1, Beta1, opt_value1 = pt3.primal_dual_simplex_solver(self.z_star_n, self.matrix_B, self.matrix_N, self.x_starB, self.b, self.c, self.Beta, self.Nu, self.objective)
        # assertAlmostEqual is used for fractions and real numbers
        self.assertAlmostEqual(self.optimal_solution, opt_value1, 3)
        self.assertAlmostEqual(self.x_b_solution.all(), x_star_b1.all(), 3)
        self.assertAlmostEqual(self.z_n_solution.all(), z_starN1.all(), 3)
        self.assertAlmostEqual(self.b_solution.all(), matrix_B1.all(), 3)
        self.assertAlmostEqual(self.n_solution.all(), matrix_N1.all(), 3)
        self.assertEqual(all(self.beta_solution), all(Beta1))
        self.assertEqual(all(self.nu_solution), all(Nu1))

    def test_4_optimal_solution_crisscross(self):
        """Check optimal solution from Criss Cross Method to compare with Simplex."""
        z_starN2, x_star_b2, matrix_B2, matrix_N2, Nu2, Beta2, opt_value2 = cs.criss_cross_solver(self.z_star_n, self.matrix_B, self.matrix_N, self.x_starB, self.b, self.c, self.Beta, self.Nu, self.objective)
        # assertAlmostEqual is used for fractions and real numbers
        self.assertAlmostEqual(self.optimal_solution, opt_value2, 3)
        self.assertAlmostEqual(self.x_b_solution.all(), x_star_b2.all(), 3)
        self.assertAlmostEqual(self.z_n_solution.all(), z_starN2.all(), 3)
        self.assertAlmostEqual(self.b_solution.all(), matrix_B2.all(), 3)
        self.assertAlmostEqual(self.n_solution.all(), matrix_N2.all(), 3)
        self.assertEqual(all(self.beta_solution), all(Beta2))
        self.assertEqual(all(self.nu_solution), all(Nu2))

    # ending the test
    def tearDown(self):
        """Cleaning up after the test."""
        pass


if __name__ == '__main__':
    unittest.main()
