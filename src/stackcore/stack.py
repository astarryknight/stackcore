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
from . import visualization as vis


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

        print(f'ref_norm : {ref_norm}')
        print(f'ref_norm : {norm_mp_og}')

        for m in self.metrics:
            if(m['type']=='Angular'):
                og_metrics.append( np.arccos(ref_norm@norm_mp_og/np.linalg.norm(norm_mp_og)) )
            elif(m['type']=="Linear"):
                og_metrics.append(
                    matrix.distance_between_planes_general(mp_og, r)
                )#calculate distance between mp_og and r
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
            mp = [mp1, mp2, mp3]
            norm_mp = matrix.get_normal(mp1, mp2, mp3)
            for k in range(len(self.metrics)):
                if(m['type']=='Angular'):
                    computed_metrics[k][i] = np.arccos(ref_norm@norm_mp)
                elif(m['type']=="Linear"):
                    computed_metrics[k][i] = matrix.distance_between_planes_general(mp, r)


        # Differences with the original points
        for i in range(len(self.metrics)):
            if(m['type']=='Angular'):
                self.delta_metrics.append( np.rad2deg(computed_metrics[i] - og_metrics[i]) )
            elif(m['type']=='Linear'):
                self.delta_metrics.append( computed_metrics[i] - og_metrics[i] )

        stop = timeit.default_timer()

        if self.save:
            plot_histogram(self.metrics, self.delta_metrics, self.path)

        print('Time: ', stop-start)
        return stop-start


## PARALLEL IMPLEMENTATION ##

@njit(parallel=True)
def loop(ncases, mp_og, tolerance_types, tolerance_ranges, tolerance_axes, tolerance_counts, component_planes, metric_refs, metric_norms, metric_type):
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

        # # Get the metrics
        mp = [mp1, mp2, mp3]
        norm_mp = nbm.get_normal_numba(mp1, mp2, mp3)
        for k in range(len(metric_refs)):
            if(metric_type=='Angular'):
                computed_metrics[k][i] = np.arccos(metric_norms[k]@norm_mp)
            elif(metric_type=="Linear"):
                computed_metrics[k][i] = nbm.distance_between_planes_numba(mp, metric_refs[k])

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
        r=self.rp
        ref_norm = matrix.get_normal(r[0], r[1], r[2])
        metric_refs = np.array([r])
        metric_norms = np.array([ref_norm])
        metric_type = self.metrics[0]['type']
        
        return (tolerance_types, tolerance_ranges, tolerance_axes, tolerance_counts, 
                component_planes, metric_refs, metric_norms, metric_type)


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
            if(m['type']=='Angular'):
                og_metrics.append( np.arccos(ref_norm@norm_mp_og/np.linalg.norm(norm_mp_og)) )
            elif(m['type']=="Linear"):
                #calculate distance between mp_og and r
                og_metrics.append(
                    nbm.distance_between_planes_numba(mp_og, r)
                )

        (tolerance_types, tolerance_ranges, tolerance_axes, tolerance_counts, 
         component_planes, metric_refs, metric_norms, metric_type) = self._prepare_numba_data()

        computed_metrics=loop(ncases, mp_og, tolerance_types, tolerance_ranges, tolerance_axes, 
                              tolerance_counts, component_planes, metric_refs, metric_norms, metric_type)
        
        # Differences with the original points
        for i in range(len(self.metrics)):
            if(metric_type=="Angular"):
                self.delta_metrics.append( np.rad2deg(computed_metrics[i]-og_metrics[i]) )
            elif(metric_type=="Linear"):
                self.delta_metrics.append( computed_metrics[i] - og_metrics[i] )


        stop = timeit.default_timer()

        if self.save: 
            print(self.delta_metrics)
            plot_histogram(self.metrics, self.delta_metrics, self.path)

        print('Time: ', stop-start)
        return stop-start


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
        self.delta_metrics = []
        self.path = path
        '''Path to save figures.'''
        self.save = save
        '''Save figure?'''

    def stack(self):
        """Stacks up all metrology tolerances to observe the effect of the as-built components."""
        og_metrics=[]
        computed_metrics=[]

        #original main plane
        mp_og = self.mp

        norm_mp_og = matrix.get_normal(mp_og[0], mp_og[1], mp_og[2])

        r=self.rp
        ref_norm = matrix.get_normal(r[0], r[1], r[2])

        for m in self.metrics:
            if(m['type']=='Angular'):
                og_metrics.append( np.arccos(ref_norm@norm_mp_og/np.linalg.norm(norm_mp_og)) )
            elif(m['type']=='Linear'):
                og_metrics.append(
                    matrix.distance_between_planes_general(mp_og, r)
                )

        # mp1 = mp_og[0] #main_plane p1
        # mp2 = mp_og[1] #main_plane p2
        # mp3 = mp_og[2] #main_plane p3
        
        # for j in range(len(self.components)):
        #     # Get the original point. The transformation matrix is applied first if it is not the first component
        #     if j == 0:
        #         pog = np.asarray([
        #             self.components[0]['points'][0]['coordinates'],
        #             self.components[0]['points'][1]['coordinates'],
        #             self.components[0]['points'][2]['coordinates']
        #         ])
        #     else:
        #         pog = np.array([(T@np.append(pog[0,:],1))[0:3],
        #                         (T@np.append(pog[1,:],1))[0:3],
        #                         (T@np.append(pog[2,:],1))[0:3]])
        #     p1 = pog[0,:].copy(); p2 = pog[1,:].copy(); p3 = pog[2,:].copy()
            
        #     p = self.components[j]['points']

        #     # Applying tolerances
        #     p1 += [p[0]['dx'], p[0]['dy'], p[0]['dz']]
        #     p2 += [p[1]['dx'], p[1]['dy'], p[1]['dz']]
        #     p3 += [p[2]['dx'], p[2]['dy'], p[2]['dz']]
        #     #does this work for a whole stack?????

        #     # Get the transformation matrix (T) related to the applied tolerance
        #     p = np.array([p1, p2, p3])
        #     T = matrix.get_transformation_matrix(pog, p)

        #     # Apply it to the mirror
        #     mp1 = (T@np.append(mp1,1))[0:3]
        #     mp2 = (T@np.append(mp2,1))[0:3]
        #     mp3 = (T@np.append(mp3,1))[0:3]

        # Start with the original main_plane points
        mp1 = self.mp[0].copy()
        mp2 = self.mp[1].copy()
        mp3 = self.mp[2].copy()

        for j in range(len(self.components)):
            # Get the original (CAD nominal) triangle for this component
            pog = np.asarray([
                self.components[j]['points'][0]['coordinates'],
                self.components[j]['points'][1]['coordinates'],
                self.components[j]['points'][2]['coordinates']
            ])

            # Apply the as-built deltas to create the measured triangle
            p1 = pog[0] + np.array([self.components[j]['points'][0]['dx'],
                                    self.components[j]['points'][0]['dy'],
                                    self.components[j]['points'][0]['dz']])
            p2 = pog[1] + np.array([self.components[j]['points'][1]['dx'],
                                    self.components[j]['points'][1]['dy'],
                                    self.components[j]['points'][1]['dz']])
            p3 = pog[2] + np.array([self.components[j]['points'][2]['dx'],
                                    self.components[j]['points'][2]['dy'],
                                    self.components[j]['points'][2]['dz']])
            
            # Build the transformation matrix from nominal (pog) → as-built (p)
            T = matrix.get_transformation_matrix(pog, [p1, p2, p3])
            
            # Apply this single transform to the current main plane
            mp1 = (T @ np.append(mp1, 1))[:3]
            mp2 = (T @ np.append(mp2, 1))[:3]
            mp3 = (T @ np.append(mp3, 1))[:3]

        # Get the angles
        mp = [mp1, mp2, mp3]
        norm_mp = matrix.get_normal(mp1, mp2, mp3)
        for m in self.metrics:
            if(m['type']=='Angular'):
                computed_metrics.append( np.arccos(ref_norm@norm_mp) )
            elif(m['type']=='Linear'):
                computed_metrics.append( matrix.distance_between_planes_general(mp, r) )

        # Differences with the original points
        for i in range(len(self.metrics)):
            if(self.metrics[i]['type']=='Angular'):
                self.delta_metrics.append( np.rad2deg(computed_metrics[i]-og_metrics[i]) )
            elif(self.metrics[i]['type']=='Linear'):
                self.delta_metrics.append( computed_metrics[i]-og_metrics[i] )

        self.visualize(mp)

        return self.delta_metrics
    
    def visualize(self, new_plane):

        self.mp
        self.rp

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        col=['red', 'blue', 'green', 'pink']

        i=0
        #for i in range(3):
        print(f"{self.mp[i][0]}, {self.mp[i][1]}, {self.mp[i][2]}")
        

        a, b, c, d = vis.get_plane_equation(self.mp)
        self.mp = np.asarray(self.mp)
        x = np.linspace(self.mp[:,0].min(), self.mp[:,0].max(), 5)
        y = np.linspace(self.mp[:,1].min(), self.mp[:,1].max(), 5)
        X, Y = np.meshgrid(x, y)
        if(c != 0):
            Z = (d - a * X - b * Y) / c
        else:
            Z = (d - a * X - b * Y)
        
        ax.plot_surface(X, Y, Z, alpha=0.75, color=col[i])
        ax.plot(*zip(self.mp[0], self.mp[1], self.mp[2]), color="red", marker='o')
            
        ax.view_init(0, 22)
        plt.tight_layout()
        plt.show()
        ax.legend()
        plt.savefig(self.path)



def plot_histogram(metrics, delta_metrics, path):
    """Plot Monte Carlo simulation deltas with normalized curve."""
    #plt.rcParams.update({'font.size':25})
    plt.rcParams['font.size'] = 17  # Set to your desired font size
    print(len(metrics))
    print(delta_metrics)
    for i in range(len(metrics)):
        d=delta_metrics[i]
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
        ax.set_xlabel(f"delta {metrics[i]['name']} "+ ('[deg]' if metrics[i]['type']=='Angular' else '[mm]')) #may change if not angle...
        ax.set_ylabel('# cases')
        plt.tight_layout()
        plt.savefig(f"{path}/delta_{metrics[i]['name']}_dist.png")