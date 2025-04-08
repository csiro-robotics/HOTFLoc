"""
Plotting recall curves for evaluated models
"""
import argparse
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

CMAP = 'tab10'
SMALL_SIZE = 16
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
BIGGER_SIZE = 20

sns.set_palette(CMAP)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def scale_array(arr: np.ndarray, new_min, new_max, old_min=None, old_max=None):
    # Find the minimum and maximum of the original array
    if old_min is None:
        old_min = arr.min()
    if old_max is None:
        old_max = arr.max()
    
    # Scale the array to the new range
    scaled_arr = (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    
    return scaled_arr

def main():

    # # Results stored as [CSWP (Unseen), CSWP (Baseline), WP, Oxford, US, RA, BD, CSC3D]
    # hotformerloc_ar1 = np.asarray([61.2, 59.8, 74.8, 96.4, 92.3, 89.2, 90.4, 80.9])
    # crossloc_ar1 =     np.asarray([10.3,  9.8,  0.0, 94.4, 82.5, 78.9, 80.5, 74.1])
    # minkloc3dv2_ar1 =  np.asarray([49.7, 54.3, 71.8, 96.3, 90.9, 86.5, 86.3, 67.1])
    # logg3dnet_ar1 =    np.asarray([25.0, 29.4, 77.3,  0.0,  0.0,  0.0,  0.0,  0.0])  # epoch 3, 4 gpu, voxel 0.5, dim 32, 9 negs
    # transloc3d_ar1 =   np.asarray([ 0.0,  0.0, 48.2, 95.0, 88.0, 82.0, 82.3, 58.2])
    # pptnet_ar1 =       np.asarray([ 0.0,  0.0,  0.0, 93.5, 90.1, 84.1, 84.6,  0.0])
    # pnvlad_ar1 =       np.asarray([ 0.0,  0.0,  0.0, 62.8, 63.2, 56.1, 57.2, 35.6])

    # Results stored as [CSWP (Unseen), CSWP (Baseline), WP, Oxford, US, RA, BD, CSC3D] (WITH CAMPUS3D AERIAL-ONLY RESULTS)
    hotformerloc_ar1 = np.asarray([61.2, 59.8, 74.8, 96.4, 92.3, 89.2, 90.4, 80.4])
    crossloc_ar1 =     np.asarray([10.3,  9.8,  0.0, 94.4, 82.5, 78.9, 80.5, 70.7])
    minkloc3dv2_ar1 =  np.asarray([49.7, 54.3, 71.8, 96.3, 90.9, 86.5, 86.3, 52.5])
    logg3dnet_ar1 =    np.asarray([25.0, 29.4, 77.3,  0.0,  0.0,  0.0,  0.0,  0.0])  # epoch 3, 4 gpu, voxel 0.5, dim 32, 9 negs
    transloc3d_ar1 =   np.asarray([ 0.0,  0.0, 48.2, 95.0, 88.0, 82.0, 82.3, 43.0])
    pptnet_ar1 =       np.asarray([ 0.0,  0.0,  0.0, 93.5, 90.1, 84.1, 84.6,  0.0])
    pnvlad_ar1 =       np.asarray([ 0.0,  0.0,  0.0, 62.8, 63.2, 56.1, 57.2, 19.1])
    
    results_ar1 = {
        # "PNVLAD" : pnvlad_ar1,
        "TransLoc3D" : transloc3d_ar1.copy(),
        "PPT-Net" : pptnet_ar1.copy(),
        "CrossLoc3D" : crossloc_ar1.copy(),
        "MinkLoc3Dv2" : minkloc3dv2_ar1.copy(),
        # "LoGG3D-Net" : logg3dnet_ar1.copy(),
        "HOTFormerLoc" : hotformerloc_ar1.copy(),
    }
    dataset_names = ['CS-Wild-Places\n(Unseen)', 'CS-Wild-Places\n(Baseline)', 'Wild-Places\n[26]',
                     'Oxford\nRobotcar [33]', 'University\nSector [48]', 'Residential\nArea [48]',
                     'Business\nDistrict [48]', 'CS-Campus3D\n[13]']
    # dataset_min = [5, 5, 47, 34-4, 61-4, 61-4, 54-4, 55-4] # w/ PNVLAD
    # dataset_min = [4.5, 5, 38, 80, 68, 68, 68, 48]  # standard
    # dataset_min = [0, 0, 38, 83, 72, 72, 68, 48]  # standard (10% below min)
    dataset_min = [0, 0, 38, 83, 72, 72, 68, 30]  # standard (10% below min) - campus3d aerial-only
    # dataset_min = [4.5, 5, 38, 80, 68, 68, 68, 35]  # campus3d scale reduced
    # dataset_min = [0]*len(dataset_names) 
    # dataset_min = [40]*len(dataset_names)
    
    # dataset_max = [97]*len(dataset_names)
    dataset_max = [None]*len(dataset_names)
    
    # dataset_rmax = [0.68, 0.65, 0.78, 0.85, 0.84, 0.82, 0.84, 0.77] # diff max on graph
    # dataset_rmax = [0.612, 0.598, 0.748, 0.964, 0.923, 0.892, 0.904, 0.809] # actual max
    # dataset_rmax = [0.82]*len(dataset_names)
    dataset_rmax = [0.80]*len(dataset_names)

    rmin = 0
    rmax = 1
    # Normalize each dataset into new range
    for i in range(len(dataset_names)):
        results_i = []
        # Get results from all models for current dataset
        for model_results in results_ar1.values():
            results_i.append(model_results[i])
        results_i = np.asarray(results_i)
        # Normalise over values
        results_i_norm = scale_array(results_i, new_min=rmin, new_max=dataset_rmax[i],
                                     old_min=dataset_min[i], old_max=dataset_max[i])
        # Return to each model dict
        for j, model_name in enumerate(results_ar1.keys()):
            results_ar1[model_name][i] = results_i_norm[j]

    # Clip AR@1 values to min range
    for model_name, model_results in results_ar1.items():
        results_ar1[model_name] = np.clip(model_results, rmin, rmax)
    
    # theta = np.deg2rad(np.asarray([0, 1, 2, 3, 0]) * 90.0)
    # theta = np.deg2rad(np.asarray(list(range(len(dataset_names))) + [0]) * 90.0)
    
    
    #### Create figure ####
    N = len(dataset_names)
    theta = radar_factory(N, frame='polygon')    
    fig, ax = plt.subplots(figsize=[6.8,6.5], subplot_kw={'projection': 'radar'})
    # ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set(xticks=theta, xticklabels=dataset_names)
    ax.xaxis.set_tick_params(rotation='auto')
    ax.set(yticks=[0.2, 0.4, 0.6, 0.8], yticklabels=[]) # remove ticklabels
    ax.set_ylim([rmin,rmax])

    # Plot radar chart
    for model_name, model_results in results_ar1.items():
        print(model_name)
        print(model_results)
        ax.plot(theta, model_results, '-', lw=2, markersize=5, label=model_name)
        ax.fill(theta, model_results, alpha=0.25, label='_nolegend_')
    # ax.set_varlabels(dataset_names)

    # Plot HOTFormerLoc values
    for ii, (ti, di) in enumerate(zip(theta, results_ar1['HOTFormerLoc'])):
        if ii == 2:  # fix leftmost point
            ti += 0.06
            di += 0.01
        elif ii == 6:  # fix leftmost point
            ti -= 0.06
            di += 0.01
        di += 0.08
        ax.text(ti, di, hotformerloc_ar1[ii], color=f'C{len(results_ar1)-1}', ha='center', va='center')
    ax.scatter(theta, results_ar1['HOTFormerLoc'], color=f'C{len(results_ar1)-1}', s=15)
    
    ax.grid(True, linestyle='--')
    ax.set_axisbelow(True)


    # # Add legend relative to top-left plot
    # legend = ax.legend(loc=(0.9, .9), labelspacing=0.1, fontsize='small')
    # legend = ax.legend(loc=(0.9, .9))

    # # OLD LEGEND:
    # angle = np.deg2rad(-15.0)
    # ax.legend(loc="upper left",
    #           bbox_to_anchor=(.46 + np.cos(angle)/2, .5 + np.sin(angle)/2))

    
    

    # ax.set(xticks=theta[:-1], xticklabels=dataset_names)
    # # ax.xaxis.set_tick_params(pad=30)
    # ax.xaxis.set_tick_params(rotation='auto')
    # # ax.set(yticks=[20, 40, 60, 80], yticklabels=['20%', '40%', '60%', '80%']) # Less radial ticks
    # ax.set(yticks=[0.2, 0.4, 0.6, 0.8], yticklabels=[]) # remove ticklabels
    # ax.set_ylim([rmin,rmax])
    # ax.set_rlabel_position(-45.0)  # Move radial labels away from plotted line
    # ax.grid(True, linestyle='--')
    # ax.set_axisbelow(True)

    # for model_name, model_results in results_ar1.items():
    #     print(model_name)
    #     print(model_results)
    #     ax.plot(theta, model_results, '-', lw=2, markersize=5, label=model_name)
    #     ax.fill_between(theta, 0, model_results, alpha=0.2)
    #     # break

    # angle = np.deg2rad(35.0)
    # ax.legend(loc="lower left",
    #           bbox_to_anchor=(.5 + np.cos(angle)/2, .5 + np.sin(angle)/2))

    
    fig.tight_layout()
    # plt.savefig('/home/gri317/work/HOT-Net/media/radar_plot.pdf', backend='pgf')
    # plt.savefig('/home/gri317/work/HOT-Net/media/radar_plot.pdf')
    # plt.savefig('/home/gri317/work/HOT-Net/media/radar_plot.svg')
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot recall curves')
    # parser.add_argument('--dataset_root', type=str, required=False, default='.')
    args = parser.parse_args()
    main()