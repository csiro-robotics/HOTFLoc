"""
Plotting recall curves for evaluated models
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title

def main():
    # (Kara + Venman) / 2
    crossloc_Baseline30m = (
        np.array([ 7.86304563, 11.63535789, 15.29661069, 17.92219469, 20.00570963, 21.67967658,
                  23.11232241, 24.41355759, 25.43308409, 26.399028,   27.32677793, 28.29649264,
                  29.24812133, 29.95869336, 30.77731477, 31.39302878, 32.01543753, 32.72490513,
                  33.14863843, 33.61893564, 34.20595743, 34.82364392, 35.4284075,  35.92616435,
                  36.66893317])
        + np.array([11.72041066, 17.60782778, 22.16803409, 25.62973858, 28.05392736, 30.31554398,
                    32.79202474, 34.90153257, 36.63414958, 38.55533593, 40.11303007, 41.69700825,
                    43.26977761, 44.8565622,  46.27292079, 47.52579515, 48.89224709, 50.01541645,
                    51.10261315, 52.06715666, 52.98948786, 54.09559191, 55.10676698, 55.86827957,
                    56.60179842])
    ) / 2
    minkloc3dv2_Baseline30m = (
        np.array([42.32591589, 49.12371355, 52.45129073, 55.14777442, 57.43829313, 59.48931798,
                  61.32172404, 63.27741674, 65.05454154, 66.5079974,  67.67073826, 68.63569012,
                  69.59765579, 70.69115436, 71.60190152, 72.39678936, 73.11882033, 73.77157119,
                  74.43107815, 74.96931052, 75.36783587, 75.84302584, 76.36436694, 76.82972667,
                  77.27384005])                               
        + np.array([66.35363681, 72.89092064, 76.27542503, 78.82626225, 81.01314222, 82.76722578,
                    84.2107221, 85.7459385,  86.93705792, 88.19393465, 89.09742106, 90.02262847,
                    90.73961025, 91.62050459, 92.22627115, 92.62150798, 93.36794535, 93.79541833,
                    94.13532102, 94.37859283, 94.69787123, 94.98972798, 95.18527514, 95.45395268,
                    95.64238122])
    ) / 2
    # NOTE: BELOW IS FROM: epoch 3, 4 gpu, voxel 0.5, dim 32, 9 negs
    logg3d_Baseline30m = (
        np.array([33.49012554, 40.53601295, 44.73137003, 47.86542657, 50.28265631, 52.19283399,
                  53.78749203, 55.1832025,  56.66595605, 57.78475464, 58.72649132, 59.81687103,
                  60.88837536, 61.75433752, 62.44262465, 63.18547154, 63.80408832, 64.65226999,
                  65.27808047, 65.7413387,  66.21057207, 66.72266533, 67.09957891, 67.51436666,
                  67.8292552 ])
        + np.array([25.25579026, 34.50205398, 40.1709577,  43.79747956, 47.5469127,  50.32992722,
                    52.60656015, 54.78144806, 56.88985918, 59.07618211, 60.73031159, 62.30306952,
                    63.45958615, 64.36391118, 65.38625539, 66.5258475,  67.45926662, 68.12283025,
                    68.93008829, 69.72363007, 70.45640294, 71.13603443, 71.73628863, 72.34222907,
                    72.9735271 ])
    ) / 2
    hotformerloc_Baseline30m = (
        np.array([64.81536431, 72.60017846, 76.08853039, 78.65307282, 80.5105329,  81.97674035,
                  83.24640715, 84.33338302, 85.32101678, 86.28640624, 86.94953556, 87.57970288,
                  88.25034572, 88.80109499, 89.39217046, 89.89792316, 90.29324799, 90.65474192,
                  90.85558476, 91.17601712, 91.4284696,  91.66257849, 91.87607178, 92.1581513,
                  92.43251873])
        + np.array([54.81249981, 62.39990492, 66.45638488, 69.12688116, 71.26906425, 72.97084077,
                    74.39591699, 75.79959712, 77.13594379, 78.02214352, 79.02292554, 79.8283265,
                    80.69700676, 81.620191,   82.42188491, 83.08370214, 83.64352354, 84.33044688,
                    84.86565667, 85.35588865, 85.80659263, 86.31542936, 86.78624807, 87.38013326,
                    87.66317313])
    ) / 2

    # (QCAT + Samford) / 2
    crossloc_Unseen30m = (
        np.array([16.48793566, 22.38605898, 28.41823056, 32.97587131, 37.93565684, 41.68900804,
                  46.38069705, 49.73190349, 53.35120643, 55.76407507, 57.90884718, 60.0536193,
                  62.19839142, 63.13672922, 65.54959786, 66.62198391, 68.49865952, 69.57104558,
                  71.04557641, 71.84986595, 72.78820375, 73.99463807, 75.06702413, 76.67560322,
                  77.61394102])
        + np.array([ 4.01968827,  6.4807219,   8.69565217, 10.33634126, 11.64889253, 12.55127153,
                    13.45365053, 14.35602953, 15.09433962, 15.99671862, 16.48892535, 17.06316653,
                    17.88351107, 18.12961444, 18.70385562, 19.44216571, 20.01640689, 20.34454471,
                    20.83675144, 21.24692371, 22.06726825, 22.39540607, 22.80557834, 23.21575062,
                    23.62592289])
    ) / 2
    minkloc3dv2_Unseen30m = (
        np.array([44.63806971, 55.36193029, 60.32171582, 66.21983914, 69.03485255, 70.91152815,
                  73.32439678, 74.66487936, 76.27345845, 77.47989276, 79.62466488, 80.16085791,
                  80.8310992,  81.90348525, 82.30563003, 82.57372654, 82.57372654, 83.24396783,
                  83.78016086, 84.18230563, 85.25469169, 86.59517426, 87.13136729, 87.80160858,
                  88.20375335])
        + np.array([54.79901559, 63.41263331, 68.00656276, 70.54963084, 73.25676784, 74.97949139,
                    75.79983593, 77.1123872,  77.85069729, 79.16324856, 80.39376538, 81.21410993,
                    82.19852338, 82.5266612,  83.01886792, 83.4290402,  83.83921247, 84.49548811,
                    84.90566038, 85.56193601, 85.89007383, 85.97210829, 86.62838392, 86.95652174,
                    87.36669401])
    ) / 2
    # NOTE: BELOW IS FROM: epoch 3, 4 gpu, voxel 0.5, dim 32, 9 negs
    logg3d_Unseen30m = (
        np.array([37.66756032, 48.25737265, 54.28954424, 59.38337802, 61.26005362, 63.40482574,
                  64.74530831, 66.62198391, 68.36461126, 69.97319035, 71.04557641, 72.11796247,
                  73.72654155, 74.93297587, 76.80965147, 78.55227882, 79.75871314, 80.02680965,
                  81.09919571, 81.90348525, 82.57372654, 83.91420912, 84.31635389, 85.5227882,
                  86.19302949])
        + np.array([12.30516817, 16.40689089, 19.03199344, 21.49302707, 23.46185398, 24.28219852,
                    25.51271534, 26.08695652, 27.07136998, 28.30188679, 29.20426579, 30.18867925,
                    31.1730927,  31.82936833, 33.14191961, 33.47005742, 33.96226415, 34.29040197,
                    34.53650533, 35.35684988, 35.93109106, 36.17719442, 36.50533224, 36.99753897,
                    37.40771124])
    ) / 2
    hotformerloc_Unseen30m = (
        np.array([47.1849866,  59.65147453, 65.01340483, 67.56032172, 70.24128686, 71.98391421,
                  74.12868633, 76.67560322, 78.82037534, 80.56300268, 81.63538874, 83.51206434,
                  84.45040214, 85.12064343, 85.92493298, 87.66756032, 88.60589812, 89.14209115,
                  89.41018767, 89.81233244, 90.08042895, 90.75067024, 91.01876676, 91.42091153,
                  91.8230563])
        + np.array([75.22559475, 81.37817884, 84.08531583, 86.05414274, 87.12059065, 88.51517637,
                    89.17145201, 89.9097621,  90.73010664, 91.30434783, 92.12469237, 92.78096801,
                    93.02707137, 93.27317473, 93.27317473, 93.43724364, 93.76538146, 94.09351928,
                    94.33962264, 94.585726,   94.66776046, 94.91386382, 94.91386382, 94.99589828,
                    94.99589828])
    ) / 2
    
    colours = ['C0-D', 'C1-o', 'C2-s', 'C3-^']


    resultsBaseline30m = {
                          u'CrossLoc3D$^\u2020$' : [crossloc_Baseline30m, colours[0]],
                          'MinkLoc3Dv2' : [minkloc3dv2_Baseline30m, colours[1]],
                          'LoGG3D-Net' : [logg3d_Baseline30m, colours[2]],
                          'HOTFormerLoc' : [hotformerloc_Baseline30m, colours[3]],
                         }
    resultsUnseen30m =  {
                         u'CrossLoc3D$^\u2020$' : [crossloc_Unseen30m, colours[0]],
                         'MinkLoc3Dv2' : [minkloc3dv2_Unseen30m, colours[1]],
                         'LoGG3D-Net' : [logg3d_Unseen30m, colours[2]],
                         'HOTFormerLoc' : [hotformerloc_Unseen30m, colours[3]],
                        }
    
    # Create x steps
    x_lim = 25
    x = [x for x in range(1, x_lim+1)]
    step_size = 5
    l = np.arange(5, x_lim+1, step_size)
    
    # split = ['Beetaloo', 'Karawatha', 'QCAT', 'Samford', 'Robson', 'Oxford (urban baseline)', 'CS-Campus3D (urban aerial baseline)']
    # split = ['Karawatha', 'QCAT', 'Samford', 'Robson', 'Oxford (urban baseline)', 'CS-Campus3D (urban aerial baseline)']
    split = ['Baseline', 'Unseen']
    # Create figure
    fig = plt.figure(figsize=[8,4.5])
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    # ax3 = fig.add_subplot(2,3,3)
    # ax4 = fig.add_subplot(2,3,4)
    # ax5 = fig.add_subplot(2,3,5)
    # ax6 = fig.add_subplot(2,3,6)
    # ax7 = fig.add_subplot(2,3,7)
    # axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7]
    # axs = [ax1,ax2,ax3,ax4,ax5,ax6]
    axs = [ax1,ax2]
    # fig.suptitle('Recall@N for 30m Threshold')
    
    for i, ax in enumerate(axs):        
        ax.set(xticks=l, xticklabels=l)
        ax.set_title(f'{split[i]}')
        ax.set_xlim([1,x_lim])
        ax.set_ylim([10,100])
        ax.set_xlabel('N - Number of Candidates')
        ax.set(yticks=np.arange(10, 100.1, 10))
        if i == 0:
            ax.set_ylabel('Average Recall @N (%)')
        # ax.grid()
    
    for labelname, result in resultsBaseline30m.items():
        ax1.plot(x, result[0][:x_lim], result[1], label=labelname, markersize=5)

    for labelname, result in resultsUnseen30m.items():
        ax2.plot(x, result[0][:x_lim], result[1], label=labelname, markersize=5)
        
    # for labelname, result in resultsQCAT30m.items():
    #     ax2.plot(x, result[0][:x_lim], result[1], label=labelname)
    
    # for labelname, result in resultsSamford30m.items():
    #     ax3.plot(x, result[0][:x_lim], result[1], label=labelname)
    
    # for labelname, result in resultsRobson30m.items():
    #     ax4.plot(x, result[0][:x_lim], result[1], label=labelname)
    
    # for labelname, result in resultsOxford30m.items():
    #     ax5.plot(x, result[0][:x_lim], result[1], label=labelname)
        
    # for labelname, result in resultsCampus30m.items():
    #     ax6.plot(x, result[0][:x_lim], result[1], label=labelname)
        
    # for ax in axs:
        # ax.legend(loc='lower right')
    ax2.legend(loc='lower right')
    
    fig.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot recall curves')
    # parser.add_argument('--dataset_root', type=str, required=False, default='.')
    args = parser.parse_args()
    main()