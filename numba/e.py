import numpy as np
from s import Stack

m=np.array([[3,  2, -1],
            [0,  3,  3],
            [-1, 2,  4]])

r=np.array([[0,  5,  1],
            [-2, 8, -4],
            [1,  7,  3]])

metrics = [
    { 'name': 'alpha',
      'type': 'angle', 
      'ref': np.array([0,0,1])
    }
]

components = [
             {'plane': np.array([[-8, 1, 5],
                                 [-4, 7, 3],
                                 [-3, 2, 6]]),
              'tolerances': [{'type': 'cylindrical',
                              'tol': [-0.1, +0.1],
                              'axis': np.array([[0,0,1], 
                                                [0,0,1],
                                                [-.7, -0.9, 0.]])},
                              {'type': 'displacement',
                               'tol' : [-0.1, +0.1],
                               'axis': np.array([[0,0,1], 
                                                 [0,0,1],
                                                 [-.7, -0.9, 0.]])}]}
]

path = "./"

s = Stack(m, r, components, metrics, path)
ncases = int(1e4)
s.monte(ncases)