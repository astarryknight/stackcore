from stackcore.stack import Stack, PStack, MStack
import argparse
import numpy as np

m=np.array([[3,  2, -1],
            [0,  3,  3],
            [-1, 2,  4]])

r=np.array([[0,  5,  1],
            [-2, 8, -4],
            [1,  7,  3]])

metrics = [{ 
  'name': 'alpha',
  'type': 'angle'
}]

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

metrology_components = [
    {
        "name": "Component",
        "points": [
            {
                "coordinates": [
                    -20,
                    10,
                    6
                ],
                "dx": 1.0,
                "dy": -1.0,
                "dz": 0.0
            },
            {
                "coordinates": [
                    -15,
                    12,
                    5
                ],
                "dx": 0.0,
                "dy": 0.2,
                "dz": 0.3
            },
            {
                "coordinates": [
                    -3,
                    5,
                    6
                ],
                "dx": -1.0,
                "dy": 0.01,
                "dz": -0.4
            }
        ]
    }
]

path = "./"

parser = argparse.ArgumentParser(description="A script that runs a stackcore montecarlo simulation example.")

parser.add_argument("-n", "--ncases", help="Set the number of cases to run the monte carlo simulation for.")
parser.add_argument("-s", "--save_fig", action="store_true", help="Save the histogram to the specified path.")
args = parser.parse_args()

ncases = int(1e3)

if args.ncases:
    ncases=int(float(args.ncases))

#Stack with regular processing
s = Stack(m, r, components, metrics, path, args.save_fig)
s.monte(ncases)

#Stack with parallel processing
ps = PStack(m, r, components, metrics, path, args.save_fig)
ps.monte(ncases)

m = MStack(m, r, metrology_components, metrics, path, args.save_fig)
print(m.stack())