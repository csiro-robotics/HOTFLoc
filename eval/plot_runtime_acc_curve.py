"""
Hero fig for HOTFLoc++
"""
import matplotlib.pyplot as plt
import numpy as np

# # PR: (runtime seconds, recall@1 [%] - 10m)
# pr_results_init = {
#     "MinkLoc3Dv2": (0.016, 26.53),
#     "HOTFormerLoc": (0.04632, 31.78),
#     "EgoNN": (0.01989, 28.23),
#     "LoGG3D-Net": (0.058, 12.03),
#     "HOTFLoc++": (0.03636, 37.73),
# }
# pr_results_rr = {
#     "EgoNN (SGV)": (0.02108, 46.6),
#     "LoGG3D-Net (SGV)": (0.5179, 21.9),
#     # "HOTFLoc++ (SGV)": (0.05907, 66.2),
#     "HOTFLoc++ (MSGV)": (0.0697, 72.1),
# }

# PR: (runtime seconds, recall@1 [%] - 30m)
pr_results_init = {
    "MinkLoc3Dv2": (0.016, 55.475),
    "HOTFormerLoc": (0.04632, 59.575),
    "EgoNN": (0.01989, 53.925),
    "LoGG3D-Net": (0.058, 27.4),
    "HOTFLoc++": (0.03636, 66.325),
}
pr_results_rr = {
    "EgoNN (SGV)": (0.02108, 61.05),
    "LoGG3D-Net (SGV)": (0.5179, 43.375),
    # "HOTFLoc++ (SGV)": (0.05907, ),
    "HOTFLoc++ (MSGV)": (0.0697, 90.65),
}

# Success Rate: (runtime seconds, success [%])
metloc_results_init = {
    "EgoNN": (27.53/1000, 54.85),
    "LoGG3D-Net": (8920/1000, 65.325),
    "HOTFLoc++": (69.83/1000, 92.975),
}
metloc_results_rr = {
    "EgoNN (SGV)": (28.72/1000, 68.65),
    "LoGG3D-Net (SGV)": (9379.9/1000, 66.3),
    # "HOTFLoc++ (SGV)": (),
    "HOTFLoc++ (MSGV)": (103.17/1000, 95.925),
}

# Extract arrays
# runtimes_init = np.array([v[0] for v in methods_init.values()])
# recalls_init = np.array([v[1] for v in methods_init.values()])

# # Pareto front calculation (min runtime, max recall@1)
# # Sort by runtime ascending
# indices_init = np.argsort(runtimes_init)
# rts_sorted_init = runtimes_init[indices_init]
# r1_sorted_init = recalls_init[indices_init]

# pareto_rts_init = [rts_sorted_init[0]]
# pareto_acc_init = [r1_sorted_init[0]]
# for i in range(1, len(rts_sorted_init)):
#     if r1_sorted_init[i] >= pareto_acc_init[-1]:
#         pareto_rts_init.append(rts_sorted_init[i])
#         pareto_acc_init.append(r1_sorted_init[i])

# IEEE-like styling
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    # "figure.figsize": (3.5, 2.8),  # single-column width for IEEE
    "figure.figsize": (3.5, 3.6),  # single-column width for IEEE
    "axes.labelsize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# fig, ax = plt.subplots()
fig, axs = plt.subplots(2, 1, sharex='all')

markers_init = ['o', 's', '^', 'D', 'p']
colors_init  = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:pink']
markers_rr = ['^', 'D', 'p']
colors_rr  = ['tab:green', 'tab:red', 'tab:pink']
# markers = ['o', 's', '^', 'h', 'D']
# colors  = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:red']

## PR
# Initial values
for (label, (rt, acc)), m, c in zip(pr_results_init.items(), markers_init, colors_init):
    axs[0].scatter(rt, acc, marker=m, label=label, s=40, linewidths=0.8, facecolors='none', edgecolors=c)
# Re-ranked values
for (label, (rt, acc)), m, c in zip(pr_results_rr.items(), markers_rr, colors_rr):
    # axs[0].scatter(rt, acc, marker=m, label=label, s=40, linewidths=0.8, facecolors=c, edgecolors=c)
    axs[0].scatter(rt, acc, marker=m, s=40, linewidths=0.8, facecolors=c, edgecolors=c)
    if 'HOTFLoc++' in label:
        # axs[0].text(rt, acc, 'Ours', ha='left', va='bottom', fontweight='bold')
        axs[0].annotate('Ours', (rt, acc), textcoords='offset points', xytext=(6,-4), fontweight='bold')

## MetLoc
# Initial values
for (label, (rt, succ)), m, c in zip(metloc_results_init.items(), markers_rr, colors_rr):  # metloc has same num methods for init and rr
    axs[1].scatter(rt, succ, marker=m, label=label, s=40, linewidths=0.8, facecolors='none', edgecolors=c)
# Re-ranked values
for (label, (rt, succ)), m, c in zip(metloc_results_rr.items(), markers_rr, colors_rr):
    # axs[1].scatter(rt, succ, marker=m, label=label, s=40, linewidths=0.8, facecolors=c, edgecolors=c)
    axs[1].scatter(rt, succ, marker=m, s=40, linewidths=0.8, facecolors=c, edgecolors=c)
    if 'HOTFLoc++' in label:
        # axs[1].text(rt, succ, 'Ours', ha='left', va='bottom', fontweight='bold')
        axs[1].annotate('Ours', (rt, succ), textcoords='offset points', xytext=(6,-4), fontweight='bold')


# # Pareto front curve
# ax.plot(pareto_rts, pareto_acc, linestyle='--', linewidth=1)

# Plot re-ranking improvement
rr_improve_indices = []
for ii, color in enumerate(colors_init):
    if color in colors_rr:
        rr_idx = colors_rr.index(color)
        rr_improve_indices.append((ii, rr_idx))
for init_idx, rr_idx in rr_improve_indices:
    pr_init_temp = list(pr_results_init.values())[init_idx]
    pr_rr_temp = list(pr_results_rr.values())[rr_idx]
    axs[0].plot([pr_init_temp[0], pr_rr_temp[0]], [pr_init_temp[1], pr_rr_temp[1]],
                linestyle='--', linewidth=1, color=colors_rr[rr_idx])
    metloc_init_temp = list(metloc_results_init.values())[rr_idx]
    metloc_rr_temp = list(metloc_results_rr.values())[rr_idx]
    axs[1].plot([metloc_init_temp[0], metloc_rr_temp[0]], [metloc_init_temp[1], metloc_rr_temp[1]],
                linestyle='--', linewidth=1, color=colors_rr[rr_idx])

# Log scale for runtime
axs[0].set_xscale('log')
# axs[0].set_xlabel("Runtime (s)")
axs[0].set_ylabel("Recall@1 (%) - 30m")
axs[0].set_xticks([0.01, 0.1, 1, 10])
axs[0].set_yticks([20,40,60,80,100])
axs[0].grid(True, which='both', linestyle=':', linewidth=0.5)
axs[0].legend(loc="upper right")

axs[1].set_xscale('log')
axs[1].set_xlabel("Runtime (s)")
axs[1].set_ylabel("Success Rate (%)")
axs[0].set_xticks([0.01, 0.1, 1, 10])
axs[1].set_yticks([40,60,80,100])
axs[1].grid(True, which='both', linestyle=':', linewidth=0.5)
# axs[0].legend(loc="upper right")

plt.tight_layout()
plt.show()
