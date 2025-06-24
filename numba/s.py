import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy.stats as stats

# from . import matrix
# from . import point_manipulation as pm

import m as matrix
import pm

from tqdm import tqdm
import timeit

class Stack:
    '''
    Tolerance stack-up analysis tools.

    '''
    def __init__(self, main_plane: npt.NDArray, ref_plane: npt.NDArray, components: dict, metrics: dict, path: str):
        self.mp = np.asanyarray(main_plane)
        '''Main plane for computations.'''
        self.rp = np.asanyarray(ref_plane)
        '''Reference plane for calculating angles. (deprecated)'''
        self.components = components
        '''Tolerance Stack up components.'''
        self.metrics = metrics
        '''Metrics to be calculated.'''
        self.delta_metrics = []
        self.path=path
        '''Path to save figures.'''

    def _prepare_numba_data(self):
        """Prepare data structures for Numba functions."""
        # Convert components to arrays that Numba can handle
        n_components = len(self.components)
        max_tolerances = max(len(comp['tolerances']) for comp in self.components)
        
        # Create structured arrays for tolerances
        tolerance_types = np.zeros((n_components, max_tolerances), dtype=np.int32)  # 0=cylindrical, 1=displacement
        tolerance_ranges = np.zeros((n_components, max_tolerances, 2), dtype=np.float64)  # [min, max]
        tolerance_axes = np.zeros((n_components, max_tolerances, 3, 3), dtype=np.float64)  # 3 axes per tolerance
        tolerance_counts = np.zeros(n_components, dtype=np.int32)  # number of tolerances per component
        
        component_planes = np.zeros((n_components, 3, 3), dtype=np.float64)
        
        for i, comp in enumerate(self.components):
            component_planes[i] = np.asarray(comp['plane'], dtype=np.float64)
            tolerance_counts[i] = len(comp['tolerances'])
            
            for j, tol in enumerate(comp['tolerances']):
                if tol['type'] == 'cylindrical':
                    tolerance_types[i, j] = 0
                elif tol['type'] == 'displacement':
                    tolerance_types[i, j] = 1
                else:
                    raise Exception(f'Incompatible tolerance type: {tol["type"]}')
                
                tolerance_ranges[i, j] = np.asarray(tol['tol'], dtype=np.float64)
                tolerance_axes[i, j] = np.asarray(tol['axis'], dtype=np.float64)
        
        # Prepare metrics data
        metric_refs = np.zeros((len(self.metrics), 3), dtype=np.float64)
        for i, metric in enumerate(self.metrics):
            metric_refs[i] = np.asarray(metric['ref'], dtype=np.float64)
        
        return (tolerance_types, tolerance_ranges, tolerance_axes, tolerance_counts, 
                component_planes, metric_refs)


    def monte(self, ncases: int):        #TODO more robust error handling
        '''
        Computes Monte Carlo simulation for mechanical tolerances.

        '''
        og_metrics = []
        computed_metrics = []
        start = timeit.default_timer()
        #original main plane
        mp_og = self.mp

        norm_mp_og = matrix.get_normal(mp_og[0], mp_og[1], mp_og[2])

        for m in self.metrics:
            og_metrics.append( np.arccos(m["ref"]@norm_mp_og/np.linalg.norm(norm_mp_og)) )
            computed_metrics.append( np.zeros(ncases) )

        (tolerance_types, tolerance_ranges, tolerance_axes, tolerance_counts, 
         component_planes, metric_refs) = self._prepare_numba_data()
        
        print(metric_refs[0][0])


        for i in tqdm(range(ncases)):
            mp1 = mp_og[0] #main_plane p1
            mp2 = mp_og[1] #main_plane p2
            mp3 = mp_og[2] #main_plane p3
            
            for j in range(len(self.components)):
                # Get the original point. The transformation matrix is applied first if it is not the first component
                if j == 0:
                    pog = self.components[0]['plane']
                else:
                    pog = np.array([(T@np.append(pog[0,:],1))[0:3],
                                    (T@np.append(pog[1,:],1))[0:3],
                                    (T@np.append(pog[2,:],1))[0:3]])
                p1 = pog[0,:]; p2 = pog[1,:]; p3 = pog[2,:]
                
                # Apply tolerances
                for t in self.components[j]['tolerances']: 
                    if t['type'] == 'cylindrical':
                        p1 = pm.point_around_axis(p1, t['axis'][0,:], np.random.uniform(t['tol'][0], t['tol'][1]), np.random.uniform(0, 2*np.pi))
                        p2 = pm.point_around_axis(p2, t['axis'][1,:], np.random.uniform(t['tol'][0], t['tol'][1]), np.random.uniform(0, 2*np.pi))
                        p3 = pm.point_around_axis(p3, t['axis'][2,:], np.random.uniform(t['tol'][0], t['tol'][1]), np.random.uniform(0, 2*np.pi))
                    elif t['type'] == 'displacement':
                        p1 = pm.point_displacement(p1, t['axis'][0,:], np.random.uniform(t['tol'][0], t['tol'][1]))
                        p2 = pm.point_displacement(p2, t['axis'][1,:], np.random.uniform(t['tol'][0], t['tol'][1]))
                        p3 = pm.point_displacement(p3, t['axis'][2,:], np.random.uniform(t['tol'][0], t['tol'][1]))
                    else:
                        raise Exception('Incompatible tolerance type')

                # Get the transformation matrix (T) related to the applied tolerance
                p = np.array([p1, p2, p3])
                T = matrix.get_transformation_matrix(pog, p)

                # Apply it to the mirror
                mp1 = (T@np.append(mp1,1))[0:3]
                mp2 = (T@np.append(mp2,1))[0:3]
                mp3 = (T@np.append(mp3,1))[0:3]

            # Get the angles
            norm_mp = matrix.get_normal(mp1, mp2, mp3)
            for k in range(len(self.metrics)):
                computed_metrics[k][i] = np.arccos(metric_refs[k]@norm_mp)

        # Differences with the original points
        for i in range(len(self.metrics)):
            self.delta_metrics.append( np.rad2deg(computed_metrics[i]-og_metrics[i]) )

        stop = timeit.default_timer()
        print('Time: ', stop-start)

        self.plot_hist()

    
    def plot_hist(self):
        '''
        Plots Monte Carlo simulation deltas along with normalized curve.

        '''
        for i in range(len(self.metrics)):
            d=self.delta_metrics[i]
            # Get the Gaussian fit 
            mu, std = stats.norm.fit(d)

            #set up plot
            fig, ax = plt.subplots()
            y,x,_ = ax.hist(d, bins = 100)
            xmin, xmax = plt.xlim()  
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)  
            ax.axvline(x = mu-3*std, linestyle = '--', color = 'grey')
            ax.axvline(x = mu+3*std, linestyle = '--', color = 'grey')
            plt.plot(x, p/p.max()*y.max(), 'k', linewidth=3, label='Fitted normal distribution')
            ax.text(0.95, 0.95, 'std = '+str((mu+std).round(3))+'\n 3std = '+str((3*(mu+std)).round(3)), transform=ax.transAxes,
                    fontsize=12, ha='right', va='top', 
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            ax.set_xlabel(f"delta {self.metrics[i]['name']} [deg]") #may change if not angle...
            ax.set_ylabel('# cases')
            plt.tight_layout()
            plt.savefig(f"{self.path}/delta_{self.metrics[i]['name']}_dist.png")