from __future__ import print_function
import numpy as np
from numpy.testing import (TestCase,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal)
from scipy.spatial import directed_hausdorff

class TestHausdorff(TestCase):
    '''Test various properties of the directed Hausdorff code.
    Based on testing code I previously wrote for the MDA 
    library.'''
   
    def setUp(self):
        self.random_angles = np.random.random((100,)) * np.pi * 2
        self.random_columns = np.column_stack((self.random_angles,
                                               self.random_angles,
                                               np.zeros((100,))))
        self.random_columns[...,0] = np.cos(self.random_columns[...,0])
        self.random_columns[...,1] = np.sin(self.random_columns[...,1])
        self.random_columns_2 = np.column_stack((self.random_angles,
                                                 self.random_angles,
                                                 np.zeros((100,))))
        self.random_columns_2[1:,0] = np.cos(self.random_columns_2[1:,0]) * 2.0
        self.random_columns_2[1:,1] = np.sin(self.random_columns_2[1:,1]) * 2.0
        # move one point farther out so we don't have two perfect circles
        self.random_columns_2[0,0] = np.cos(self.random_columns_2[0,0]) * 3.3
        self.random_columns_2[0,1] = np.sin(self.random_columns_2[0,1]) * 3.3
        self.path_1 = self.random_columns
        self.path_2 = self.random_columns_2

    def tearDown(self):
        del self.random_angles
        del self.random_columns
        del self.random_columns_2
        del self.path_1
        del self.path_2

    def test_symmetry(self):
        '''Ensure that the directed (asymmetric)
        Hausdorff distance is actually asymmetric
        '''
        forward = directed_hausdorff(self.path_1, self.path_2)
        reverse = directed_hausdorff(self.path_2, self.path_1)
        self.assertNotEqual(forward, reverse)


