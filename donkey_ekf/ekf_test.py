from ekf import donkey_ekf
import numpy as np
import unittest

class TestEKFMethods(unittest.TestCase):

    def assert_matrix(self, matrixA, matrixB, deci=3):
        if len(matrixA.shape) == 1:
            row_size = matrixA.shape[0]
            for row in range(row_size):
                self.assertAlmostEqual(matrixA[row], matrixB[row], deci)
        else:
            row_size, col_size = matrixA.shape
            for row in range(row_size):
                for col in range(col_size):
                    self.assertAlmostEqual(matrixA[row][col], matrixB[row][col], deci)


    def test_motion_update(self):
        sigma = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0.1]])
        miu = np.array([1, 1, 0])
        delta = 0.5
        v = 1
        ekf = donkey_ekf()
        miu_bar, sigma_bar = ekf.motion_update(miu, sigma, delta, v)
        expect_miu_bar = np.array([1.09646599,
                                   1.02634981,
                                   0.17566537])
        expect_sigma_bar = np.array([[ 2.00694312, -0.0254186,  -0.02634981],
                                     [-0.0254186,   2.09305688,  0.09646599],
                                     [-0.02634981,  0.09646599,  0.2       ]])
        self.assert_matrix(miu_bar, expect_miu_bar, 3)
        self.assert_matrix(sigma_bar, expect_sigma_bar, 3)

    def test_expected_measure(self):
        landmark_list = [[5, 6],[-2, 3]]
        sigma_bar = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0.1]])
        miu_bar = np.array([1, 1, 0.5])
        Qt = np.array([[1, 0],
                       [0, 0.1]])
        ekf = donkey_ekf()
        landmark_psi_exp_z_H = ekf.expect_measure(landmark_list, miu_bar, sigma_bar, Qt)
        # first landmark
        z_bar_list = [np.array([6.4031, 0.3961]),
                      np.array([3.6056, 2.0536])]
        H_list = [np.array([[0.6247, -0.7809, 0],
                            [0.1220, 0.0976, -0.0244]]),
                  np.array([[-0.8321, -0.5547, 0],
                            [0.1538, -0.2308, -0.0769]])]
        psi_list = [np.array([[2.0, 0],
                         [0, 0.12445]]),
                    np.array([[2, 0],
                              [0, 0.17751]])]
        for i in range(2):
            self.assert_matrix(landmark_psi_exp_z_H[i].expected_z, z_bar_list[i])
            self.assert_matrix(landmark_psi_exp_z_H[i].H, H_list[i])
            self.assert_matrix(landmark_psi_exp_z_H[i].psi, psi_list[i])



if __name__ == '__main__':
    unittest.main()