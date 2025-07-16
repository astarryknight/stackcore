import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy.stats as stats
from numba import njit, prange
from tqdm import tqdm
import timeit

from . import matrix
from . import point_manipulation as pm
from . import numba_modules as nbm


class Stack:#SStack:
    """Sequential Stack for stack-up tolerance analysis."""
    def __init__(self, main_plane: npt.NDArray, ref_plane: npt.NDArray, components: dict, metrics: dict, path: str, save: bool):
        self.mp = np.asanyarray(main_plane)
        '''Main plane for computations.'''
        self.rp = np.asanyarray(ref_plane)
        '''Reference plane for computing metrics.'''
        self.components = components
        '''Tolerance Stack up components.'''
        self.metrics = metrics
        '''Metrics to be calculated.'''
        self.delta_metrics = []
        self.path = path
        '''Path to save figures.'''
        self.save = save
        '''Save figure?'''

    def monte(self, ncases: int):        #TODO more robust error handling
        """Compute Monte Carlo simulation for mechanical tolerances."""
        og_metrics = []
        computed_metrics = []
        start = timeit.default_timer()
        #original main plane
        mp_og = self.mp

        norm_mp_og = matrix.get_normal(mp_og[0], mp_og[1], mp_og[2])

        r=self.rp
        ref_norm = matrix.get_normal(r[0], r[1], r[2])

        for m in self.metrics:
            og_metrics.append( np.arccos(ref_norm@norm_mp_og/np.linalg.norm(norm_mp_og)) )
            computed_metrics.append( np.zeros(ncases) )

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
                computed_metrics[k][i] = np.arccos(ref_norm@norm_mp)

        # Differences with the original points
        for i in range(len(self.metrics)):
            self.delta_metrics.append( np.rad2deg(computed_metrics[i]-og_metrics[i]) )

        stop = timeit.default_timer()

        if self.save:
            self.plot_hist()

        print('Time: ', stop-start)
        return stop-start

    def plot_hist(self):
        """Plot Monte Carlo simulation deltas with Gaussian (Normal) distribution."""
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
            ax.text(0.95, 0.95, 'std = '+str((mu+std).round(3))+'\n 3std = '+str((3*(mu+std)).round(3))+'\nWorst: ('+str(np.round(np.min(d),2))+','+str(np.round(np.max(d),2))+')', transform=ax.transAxes,
                    fontsize=12, ha='right', va='top', 
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            ax.set_xlabel(f"delta {self.metrics[i]['name']} [deg]") #may change if not angle...
            ax.set_ylabel('# cases')
            plt.tight_layout()
            plt.savefig(f"{self.path}/delta_{self.metrics[i]['name']}_dist.png")


## PARALLEL IMPLEMENTATION ##

@njit(parallel=True)
def loop(ncases, mp_og, tolerance_types, tolerance_ranges, tolerance_axes, tolerance_counts, component_planes, metric_refs):
    """Parallel loop for the Monte Carlo simulation."""
    n_metrics = metric_refs.shape[0]
    computed_metrics = np.zeros((n_metrics, ncases), dtype=np.float64)

    for i in prange(ncases):
        mp1 = mp_og[0].copy() #main_plane p1
        mp2 = mp_og[1].copy() #main_plane p2
        mp3 = mp_og[2].copy() #main_plane p3
        
        for j in range(len(component_planes)):
            # Get the original point. The transformation matrix is applied first if it is not the first component
            pog = component_planes[j].copy()
            p1, p2, p3 = pog[0].copy(), pog[1].copy(), pog[2].copy() # may not work properly for more than one component... but we'll see 
            
            # Apply tolerances
            for k in range(tolerance_counts[j]): 
                if tolerance_types[j][k] == 0:
                    p1 = nbm.point_around_axis_numba(p1, tolerance_axes[j][k][0,:], np.random.uniform(tolerance_ranges[j][k][0], tolerance_ranges[j][k][1]), np.random.uniform(0, 2*np.pi))
                    p2 = nbm.point_around_axis_numba(p2, tolerance_axes[j][k][1,:], np.random.uniform(tolerance_ranges[j][k][0], tolerance_ranges[j][k][1]), np.random.uniform(0, 2*np.pi))
                    p3 = nbm.point_around_axis_numba(p3, tolerance_axes[j][k][2,:], np.random.uniform(tolerance_ranges[j][k][0], tolerance_ranges[j][k][1]), np.random.uniform(0, 2*np.pi))
                elif tolerance_types[j][k] == 1:
                    p1 = nbm.point_displacement(p1, tolerance_axes[j][k][0,:], np.random.uniform(tolerance_ranges[j][k][0], tolerance_ranges[j][k][1]))
                    p2 = nbm.point_displacement(p2, tolerance_axes[j][k][1,:], np.random.uniform(tolerance_ranges[j][k][0], tolerance_ranges[j][k][1]))
                    p3 = nbm.point_displacement(p3, tolerance_axes[j][k][2,:], np.random.uniform(tolerance_ranges[j][k][0], tolerance_ranges[j][k][1]))
                else:
                    #raise Exception('Incompatible tolerance type')
                    continue #raise makes a lot of problems in numba

            # Get the transformation matrix (T) related to the applied tolerance
            p_new = np.zeros((3, 3), dtype=np.float64)
            p_new[0], p_new[1], p_new[2] = p1, p2, p3

            T = nbm.get_transformation_matrix(pog, p_new)

            # Apply it to the mirror
            mp1_homogeneous = np.array([mp1[0], mp1[1], mp1[2], 1.0])
            mp2_homogeneous = np.array([mp2[0], mp2[1], mp2[2], 1.0])
            mp3_homogeneous = np.array([mp3[0], mp3[1], mp3[2], 1.0])
            
            mp1 = nbm.matrix_vector_multiply_3x4(T, mp1_homogeneous)
            mp2 = nbm.matrix_vector_multiply_3x4(T, mp2_homogeneous)
            mp3 = nbm.matrix_vector_multiply_3x4(T, mp3_homogeneous)

        # Get the angles
        norm_mp = nbm.get_normal_numba(mp1, mp2, mp3)
        for k in range(len(metric_refs)):
            computed_metrics[k][i] = np.arccos(metric_refs[k]@norm_mp)

    return computed_metrics


class PStack:
    """Parallel stack for stack-up tolerance analysis."""
    def __init__(self, main_plane: npt.NDArray, ref_plane: npt.NDArray, components: dict, metrics: dict, path: str, save: bool):
        self.mp = np.asanyarray(main_plane, dtype=np.float64)  # Ensure float64
        '''Main plane for computations.'''
        self.rp = np.asanyarray(ref_plane, dtype=np.float64)  # Ensure float64
        '''Reference plane for computing metrics.'''
        self.components = components
        '''Tolerance Stack up components.'''
        self.metrics = metrics
        '''Metrics to be calculated.'''
        self.delta_metrics = []
        self.path = path
        '''Path to save figures.'''
        self.save = save
        '''Save figure?'''

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
        # metric_refs = np.zeros((len(self.metrics), 3), dtype=np.float64)
        # for i, metric in enumerate(self.metrics):
        #     metric_refs[i] = np.asarray(metric['ref'], dtype=np.float64)
        r=self.rp
        ref_norm = matrix.get_normal(r[0], r[1], r[2])
        metric_refs = np.array([ref_norm])
        
        return (tolerance_types, tolerance_ranges, tolerance_axes, tolerance_counts, 
                component_planes, metric_refs)


    def monte(self, ncases: int):
        """Compute Monte Carlo simulation for mechanical tolerances."""
        og_metrics = []
        start = timeit.default_timer()
        #original main plane
        mp_og = self.mp

        norm_mp_og = matrix.get_normal(mp_og[0], mp_og[1], mp_og[2])

        r=self.rp
        ref_norm = matrix.get_normal(r[0], r[1], r[2])

        for m in self.metrics:
            og_metrics.append( np.arccos(ref_norm@norm_mp_og/np.linalg.norm(norm_mp_og)) )

        (tolerance_types, tolerance_ranges, tolerance_axes, tolerance_counts, 
         component_planes, metric_refs) = self._prepare_numba_data()

        computed_metrics=loop(ncases, mp_og, tolerance_types, tolerance_ranges, tolerance_axes, 
                              tolerance_counts, component_planes, metric_refs)
        
        # Differences with the original points
        for i in range(len(self.metrics)):
            self.delta_metrics.append( np.rad2deg(computed_metrics[i]-og_metrics[i]) )

        stop = timeit.default_timer()

        if self.save: 
            self.plot_hist()

        print('Time: ', stop-start)
        return stop-start
    
    def plot_hist(self):
        """Plot Monte Carlo simulation deltas with normalized curve."""
        # plt.rcParams.update({'font.size':18})
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
            ax.text(0.95, 0.95, 'std = '+str((mu+std).round(3))+'\n 3std = '+str((3*(mu+std)).round(3))+'\nWorst: ('+str(np.round(np.min(d),2))+','+str(np.round(np.max(d),2))+')', transform=ax.transAxes,
                    fontsize=12, ha='right', va='top', 
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            ax.set_xlabel(f"delta {self.metrics[i]['name']} [deg]") #may change if not angle...
            ax.set_ylabel('# cases')
            plt.tight_layout()
            plt.savefig(f"{self.path}/delta_{self.metrics[i]['name']}_dist.png")



## METROLOGY IMPLEMENTATION ##

class MStack:
    """Stack for using metrology data to compare CAD assemblies with as-built components."""
    def __init__(self, main_plane: npt.NDArray, ref_plane: npt.NDArray, components: dict, metrics: dict, path: str, save: bool):
        self.mp = np.asanyarray(main_plane, dtype=np.float64)  # Ensure float64
        '''Main plane for computations.'''
        self.rp = np.asanyarray(ref_plane, dtype=np.float64)  # Ensure float64
        '''Reference plane for computing metrics.'''
        self.components = components
        '''Tolerance Stack up components.'''
        self.metrics = metrics
        '''Metrics to be calculated.'''
        self.delta_metric = 0.0
        self.path = path
        '''Path to save figures.'''
        self.save = save
        '''Save figure?'''

    def stack(self):
        """Stacks up all metrology tolerances to observe the effect of the as-built components."""

        #original main plane
        mp_og = self.mp

        norm_mp_og = matrix.get_normal(mp_og[0], mp_og[1], mp_og[2])

        r=self.rp
        ref_norm = matrix.get_normal(r[0], r[1], r[2])

        og_metric = np.arccos(ref_norm@norm_mp_og/np.linalg.norm(norm_mp_og))

        mp1 = mp_og[0] #main_plane p1
        mp2 = mp_og[1] #main_plane p2
        mp3 = mp_og[2] #main_plane p3
        
        for j in range(len(self.components)):
            # Get the original point. The transformation matrix is applied first if it is not the first component
            if j == 0:
                pog = np.asarray([
                    self.components[0]['points'][0]['coordinates'],
                    self.components[0]['points'][1]['coordinates'],
                    self.components[0]['points'][2]['coordinates']
                ])
            else:
                pog = np.array([(T@np.append(pog[0,:],1))[0:3],
                                (T@np.append(pog[1,:],1))[0:3],
                                (T@np.append(pog[2,:],1))[0:3]])
            p1 = pog[0,:].copy(); p2 = pog[1,:].copy(); p3 = pog[2,:].copy()
            
            p = self.components[j]['points']

            # Applying tolerances
            p1 += [p[0]['dx'], p[0]['dy'], p[0]['dz']]
            p2 += [p[1]['dx'], p[1]['dy'], p[1]['dz']]
            p3 += [p[2]['dx'], p[2]['dy'], p[2]['dz']]
            #does this work for a whole stack?????

            # Get the transformation matrix (T) related to the applied tolerance
            p = np.array([p1, p2, p3])
            T = matrix.get_transformation_matrix(pog, p)

            # Apply it to the mirror
            mp1 = (T@np.append(mp1,1))[0:3]
            mp2 = (T@np.append(mp2,1))[0:3]
            mp3 = (T@np.append(mp3,1))[0:3]

        # Get the angles
        norm_mp = matrix.get_normal(mp1, mp2, mp3)
        computed_metric = np.arccos(ref_norm@norm_mp)

        # Differences with the original points
        self.delta_metric = ( np.rad2deg(computed_metric-og_metric) )

        return self.delta_metric
