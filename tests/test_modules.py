from stackcore import matrix
from stackcore import point_manipulation as pm
import numpy as np

def test_norm():
    points=[
        [[0.737, 0.068, 0.382], [0.828, 0.198, 0.795], [0.508, 0.019, 0.736], [0.45606074, -0.87272887,  0.17422089]], 
        [[0.282, 0.99,  0.154], [0.812, 0.667, 0.875], [0.96,  0.07,  0.744], [0.82714061,  0.30817746, -0.46996284]], 
        [[0.763, 0.501, 0.537], [0.789, 0.956, 0.461], [0.154, 0.718, 0.085], [-0.54815667,  0.16817231,  0.81929381]]
    ]
    for p in points:
        p1, p2, p3, norm = np.array(p)
        assert(np.allclose(matrix.get_normal(p1, p2, p3), norm))

def test_transform():
    planes=[
        
    ]