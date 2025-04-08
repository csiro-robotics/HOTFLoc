"""
Plotting recall curves for evaluated models
"""
import argparse
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

CMAP = 'tab10'
SMALL_SIZE = 12
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

def main():

    # Results stored as [CSWP, WP, CSC3D, Oxford, Repeat]
    hotformerloc_ar1 = np.asarray([60.5, 74.8, 80.9, 92.1, 60.5])
    crossloc_ar1 =     np.asarray([10.1,  0.0, 74.1, 84.1, 10.1])
    minkloc3dv2_ar1 =  np.asarray([52.0, 71.8, 67.1, 90.0, 52.0])
    logg3dnet_ar1 =    np.asarray([15.1, 77.3,  0.0,  0.0, 15.1])
    transloc3d_ar1 =   np.asarray([ 0.0, 48.2, 58.2, 86.5,  0.0])
    pnvlad_ar1 =       np.asarray([ 0.0,  0.0, 35.6, 59.8,  0.0])
    
    results_ar1 = {
        "PNVLAD" : pnvlad_ar1,
        "TransLoc3D" : transloc3d_ar1,
        "CrossLoc3D" : crossloc_ar1,
        "MinkLoc3Dv2" : minkloc3dv2_ar1,
        "LoGG3D-Net" : logg3dnet_ar1,
        "HOTFormerLoc (Ours)" : hotformerloc_ar1,
    }
    
    dataset_names = ['In-House', 'Wild-Places', 'CS-Campus3D', 'Oxford RobotCar']
    thetamin = 0
    thetamax = 360
    theta = np.deg2rad(np.asarray([0, 1, 2, 3, 0]) * 90.0)
    rmin = 10
    rmax = 95
    # Clip AR@1 values to min range
    for model_name, model_results in results_ar1.items():
        results_ar1[model_name] = np.clip(model_results, rmin+0.1, rmax)
    
    # thetamin = 45
    # thetamax = 180
    # theta = np.deg2rad(np.asarray([45, 90, 135, 180, 45]))
    
    # Create figure
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=[10,8])

    ax.set_thetamin(thetamin)
    ax.set_thetamax(thetamax)
    ax.set(xticks=theta[:-1], xticklabels=dataset_names)
    # ax.xaxis.set_tick_params(pad=30)
    ax.xaxis.set_tick_params(rotation='auto')
    ax.set(yticks=[20, 40, 60, 80], yticklabels=['20%', '40%', '60%', '80%']) # Less radial ticks
    ax.set_ylim([rmin,rmax])
    ax.set_rlabel_position(-45.0)  # Move radial labels away from plotted line
    ax.grid(True, linestyle='--')
    ax.set_axisbelow(True)

    for model_name, model_results in results_ar1.items():
        print(model_name)
        print(model_results)
        ax.plot(theta, model_results, '-', lw=2, markersize=5, label=model_name)
        ax.fill_between(theta, 0, model_results, alpha=0.2)
        # break

    angle = np.deg2rad(35.0)
    ax.legend(loc="lower left",
              bbox_to_anchor=(.5 + np.cos(angle)/2, .5 + np.sin(angle)/2))
    
    # ax.set(xticks=l, xticklabels=l)
    # ax.set_title(f'{split[i]}')
    # ax.set_xlim([1,x_lim])
    # ax.set_ylim([0,100])
    # ax.set_xlabel('N - Number of Candidates')
    # if i == 0:
    #     ax.set_ylabel('Average Recall @N (%)')
    
    # for labelname, result in resultsBaseline30m.items():
    #     ax1.plot(x, result[0][:x_lim], result[1], label=labelname)
    
    # ax1.legend(loc='lower right')
    
    fig.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot recall curves')
    # parser.add_argument('--dataset_root', type=str, required=False, default='.')
    args = parser.parse_args()
    main()