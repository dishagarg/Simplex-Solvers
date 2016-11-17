import unittest
import numpy as np
import project_part1 as pt1
import project_part2 as pt2


class UnitTest5(unittest.TestCase):
    """Fifth test case."""
    # preparing to test
    def setUp(self):
        """Setting up for the test."""
        # Primal Problem
        self.a = np.asarray([[1, 4, 0, 1, 0, 0], [3, -1, 1, 0, 1, 0]]).astype(np.float)
        self.b = np.asarray([[1], [3]]).astype(np.float)
        self.c = np.asarray([[4], [1], [3]]).astype(np.float)
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

        # Converted the Primal problem into Dual
        self.a1 = np.asarray([[-1, -3, 1, 0, 0], [-4, 1, 0, 1, 0], [0, -1, 0, 0, 1]]).astype(np.float)
        self.b1 = np.asarray([[-4], [-1], [-3]]).astype(np.float)
        self.c1 = np.asarray([[-1], [-3]]).astype(np.float)
        rows_b1, cols_b1 = np.shape(self.b1)
        rows_c1, cols_c1 = np.shape(self.c1)
        self.Nu1 = np.arange(1, rows_c1 + 1)
        self.Beta1 = np.arange(len(self.Nu1) + 1, rows_b1 + len(self.Nu1) + 1)
        split_a1 = np.hsplit(self.a1, [rows_c1, rows_b1 + rows_c1])
        self.matrix_N1 = split_a1[0]
        self.matrix_B1 = split_a1[1]
        self.x_starB1 = self.b1
        self.z_star_n1 = -1 * self.c1
        self.objective = 0

        # Complementary Slackness to verify
        self.beta_solution = [2, 3]
        self.nu_solution = [4, 1, 5]
        self.beta_solution_for_dual = [1, 3, 2]
        self.nu_solution_for_dual = [4, 5]

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

    def test_3_complementary_slackness(self):
        """Check for Complementary Slackness."""
        print "Solving by Primal Simplex."
        z_starN1, x_star_b1, matrix_B1, matrix_N1, Nu1, Beta1, opt_value1 = pt1.primal_simplex_solver(self.z_star_n, self.matrix_B, self.matrix_N, self.x_starB, self.b, self.c, self.Beta, self.Nu, self.objective)
        self.assertEqual(all(self.beta_solution), all(Beta1))
        self.assertEqual(all(self.nu_solution), all(Nu1))

    def test_4_complementary_slackness(self):
        """Check for Complementary Slackness."""
        print "Solving by Dual Simplex."
        z_starN1, x_star_b1, matrix_B1, matrix_N1, Nu1, Beta1, opt_value1 = pt2.dual_simplex_solver(self.x_starB1, self.matrix_B1, self.matrix_N1, self.z_star_n1, self.b1, self.c1, self.Beta1, self.Nu1, self.objective)
        self.assertEqual(all(self.beta_solution_for_dual), all(Beta1))
        self.assertEqual(all(self.nu_solution_for_dual), all(Nu1))

    # ending the test
    def tearDown(self):
        """Cleaning up after the test."""
        pass


if __name__ == '__main__':
    unittest.main()
