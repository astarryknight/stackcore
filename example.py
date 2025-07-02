from stackcore.stack import Stack, PStack
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