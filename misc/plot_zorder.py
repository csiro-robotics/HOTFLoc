"""
Create visualisations for z-order curves.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import zCurve
from misc.torch_utils import set_seed

set_seed()

if __name__ == '__main__':
    OCT_DEPTH = 3
    WINDOW_SIZE = 7
    POINT_SIZE = 20
    BG_LINE_WIDTH = 0.005
    MARKER_CONFIG = {'marker': 'o', 'markersize': 7, 'markerfacecolor': 'white'}
    QUIVER_CONFIG = {'scale_units': 'xy', 'angles': 'xy', 'scale': 1,
                     'headlength': 3, 'headaxislength': 3}
    ### Generate full grid for z-order ###
    xx, yy = np.meshgrid(np.arange(2**OCT_DEPTH), np.arange(2**OCT_DEPTH)) # full grid
    pts_fullgrid = np.stack((xx.flatten(), yy.flatten()), axis=-1)
    major_ticks = np.linspace(0, 1, 2**OCT_DEPTH + 1) 
    
    # Generate z-order curve for these (flip x and y for correct ordering)
    # z_order = zCurve.par_interlace(pts_fullgrid.tolist(), dims=2)
    z_order = zCurve.par_interlace(np.fliplr(pts_fullgrid).tolist(), dims=2)
    # Sort pts by z-order
    order_idx = np.argsort(z_order)
    pts_fullgrid_ordered = pts_fullgrid[order_idx]

    # Offset points so that grid aligns correctly
    pts_fullgrid = pts_fullgrid.astype(np.float32) + 0.5
    pts_fullgrid_ordered = pts_fullgrid_ordered.astype(np.float32) + 0.5
    # Scale to [0, 1] range
    pts_fullgrid /= 2**OCT_DEPTH
    pts_fullgrid_ordered /= 2**OCT_DEPTH
    
    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(*pts.T, c='black')
    # ax.set_aspect('equal', adjustable='box')
    # ax.set_xlim(0, 2**OCT_DEPTH)
    # ax.set_ylim(0, 2**OCT_DEPTH)
    # ax.set_xticks(major_ticks)
    # ax.set_yticks(major_ticks)
    # ax.grid()
    # ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1)
    # ax.plot(*pts_fullgrid_ordered.T, color='black', linewidth=2)
    x, y = pts_fullgrid_ordered.T
    ax.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1],
              width=BG_LINE_WIDTH, color='black', zorder=1, **QUIVER_CONFIG)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.grid()
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.tight_layout()

    ### Generate z-order windows over set of points ###
    # NUM_PTS = 50
    # pts = np.random.rand(NUM_PTS, 2) * 2**OCT_DEPTH
    # pts = (0.25 * np.random.randn(NUM_PTS, 2) + 0.5) * 2**OCT_DEPTH
    # pts = np.delete(pts, np.where((pts > 2**OCT_DEPTH) & (pts < 0)), axis=0)
    
    # pts = np.array(  # Pseudo-pointcloud (old version)
    #     [[0.16, 0.21, 0.18, 0.05, 0.18, 0.23, 0.30, 0.43, 0.48, 0.48, 0.04,
    #       0.15, 0.20, 0.23, 0.18, 0.52, 0.53, 0.60, 0.65, 0.73, 0.65, 0.66,
    #       0.58, 0.61, 0.58, 0.65, 0.67, 0.73, 0.97, 0.97, 0.97, 0.97, 0.16,
    #       0.23, 0.69, 0.42, 0.41, 0.47, 0.34, 0.44],
    #      [0.05, 0.09, 0.22, 0.43, 0.47, 0.41, 0.05, 0.48, 0.52, 0.59, 0.78,
    #       0.80, 0.85, 0.97, 0.92, 0.55, 0.18, 0.19, 0.19, 0.21, 0.30, 0.43,
    #       0.82, 0.85, 0.90, 0.93, 0.82, 0.76, 0.16, 0.28, 0.41, 0.54, 0.72,
    #       0.65, 0.07, 0.83, 0.79, 0.82, 0.85, 0.93]]
        # ).T * 2**OCT_DEPTH
    # pts = np.array(  # Pseudo-pointcloud
    #     [[0.16, 0.21, 0.14, 0.05, 0.18, 0.23, 0.30, 0.43, 0.48, 0.48, 0.82,
    #       0.15, 0.20, 0.23, 0.18, 0.52, 0.53, 0.60, 0.65, 0.73, 0.65, 0.66,
    #       0.58, 0.61, 0.58, 0.65, 0.67, 0.73, 0.90, 0.97, 0.97, 0.97, 0.16,
    #       0.23, 0.69, 0.42, 0.41, 0.47, 0.34, 0.44, 0.84, 0.98, 0.95, 0.90,
    #       0.53, 0.54, 0.58, 0.57, 0.46, 0.47, 0.84],
    #      [0.05, 0.09, 0.19, 0.43, 0.47, 0.43, 0.05, 0.48, 0.52, 0.59, 0.08,
    #       0.80, 0.85, 0.97, 0.92, 0.55, 0.18, 0.19, 0.19, 0.21, 0.30, 0.43,
    #       0.82, 0.85, 0.90, 0.93, 0.82, 0.76, 0.20, 0.27, 0.41, 0.54, 0.72,
    #       0.65, 0.07, 0.83, 0.79, 0.82, 0.85, 0.93, 0.94, 0.76, 0.85, 0.90,
    #       0.45, 0.48, 0.48, 0.47, 0.45, 0.41, 0.86]]
    #     ).T * 2**OCT_DEPTH

    # pts = np.array(  # Letter G
    #     [[0.80, 0.79, 0.74, 0.65, 0.50, 0.37, 0.25, 0.17, 0.24, 0.14, 0.21,
    #     0.18, 0.13, 0.22, 0.25, 0.16, 0.26, 0.25, 0.34, 0.46, 0.57, 0.64,
    #     0.69, 0.78, 0.77, 0.73, 0.77, 0.77, 0.71, 0.63, 0.52, 0.79, 0.80,
    #     0.65, 0.77, 0.84, 0.91, 0.84, 0.78, 0.82, 0.71, 0.75, 0.25, 0.10,
    #     0.59, 0.89, 0.73, 0.82, 0.84, 0.58],
    #     [0.86, 0.67, 0.74, 0.83, 0.86, 0.85, 0.78, 0.70, 0.64, 0.54, 0.53,
    #     0.48, 0.41, 0.36, 0.43, 0.26, 0.20, 0.27, 0.14, 0.11, 0.10, 0.13,
    #     0.17, 0.18, 0.21, 0.30, 0.37, 0.43, 0.43, 0.45, 0.46, 0.29, 0.39,
    #     0.47, 0.47, 0.47, 0.44, 0.43, 0.83, 0.76, 0.84, 0.79, 0.71, 0.50,
    #     0.47, 0.47, 0.24, 0.24, 0.35, 0.86]
    #     ]).T * 2**OCT_DEPTH
    
    pts = np.array(  # Letter S
        [[0.82, 0.16, 0.82, 0.82, 0.82, 0.79, 0.76, 0.72, 0.67, 0.61, 0.53,
          0.79, 0.78, 0.74, 0.70, 0.62, 0.55, 0.52, 0.48, 0.44, 0.42, 0.37,
          0.35, 0.38, 0.37, 0.37, 0.36, 0.37, 0.38, 0.40, 0.42, 0.45, 0.49,
          0.54, 0.60, 0.65, 0.68, 0.70, 0.72, 0.76, 0.78, 0.81, 0.83, 0.84,
          0.83, 0.82, 0.79, 0.76, 0.73, 0.69, 0.64, 0.59, 0.51, 0.44, 0.40,
          0.34, 0.29, 0.26, 0.23, 0.21, 0.17, 0.15, 0.16, 0.15, 0.16, 0.16,
          0.17, 0.20, 0.22, 0.25, 0.30, 0.36, 0.43, 0.52, 0.60, 0.68, 0.71,
          0.73, 0.71, 0.66, 0.59, 0.54, 0.41, 0.30, 0.27, 0.24, 0.22, 0.22,
          0.24, 0.28, 0.32, 0.36, 0.74, 0.76, 0.79, 0.77, 0.74, 0.70, 0.69,
          0.63, 0.55, 0.58, 0.60, 0.66, 0.65, 0.57, 0.50, 0.48, 0.41, 0.35,
          0.42, 0.40, 0.35, 0.31, 0.26, 0.31, 0.29, 0.27, 0.32, 0.18, 0.20,
          0.24, 0.27, 0.37, 0.48, 0.59, 0.67, 0.72, 0.38, 0.43, 0.49, 0.58,
          0.67, 0.70, 0.27, 0.35, 0.51, 0.38, 0.41, 0.51],
         [0.87, 0.05, 0.84, 0.79, 0.74, 0.75, 0.79, 0.82, 0.84, 0.85, 0.88,
          0.86, 0.84, 0.84, 0.85, 0.87, 0.87, 0.88, 0.87, 0.88, 0.87, 0.87,
          0.85, 0.84, 0.82, 0.81, 0.76, 0.72, 0.67, 0.64, 0.61, 0.59, 0.56,
          0.54, 0.52, 0.49, 0.48, 0.47, 0.45, 0.42, 0.40, 0.35, 0.30, 0.27,
          0.23, 0.20, 0.14, 0.12, 0.09, 0.07, 0.04, 0.03, 0.02, 0.02, 0.03,
          0.05, 0.06, 0.07, 0.07, 0.06, 0.05, 0.06, 0.09, 0.17, 0.20, 0.26,
          0.28, 0.26, 0.20, 0.16, 0.11, 0.09, 0.08, 0.07, 0.07, 0.11, 0.16,
          0.22, 0.27, 0.30, 0.32, 0.33, 0.37, 0.46, 0.50, 0.56, 0.65, 0.75,
          0.78, 0.83, 0.85, 0.87, 0.14, 0.19, 0.19, 0.29, 0.38, 0.35, 0.35,
          0.34, 0.37, 0.41, 0.43, 0.41, 0.46, 0.49, 0.49, 0.42, 0.42, 0.48,
          0.53, 0.58, 0.59, 0.56, 0.62, 0.69, 0.73, 0.78, 0.81, 0.24, 0.17,
          0.12, 0.09, 0.07, 0.05, 0.05, 0.08, 0.11, 0.89, 0.90, 0.91, 0.90,
          0.88, 0.86, 0.69, 0.65, 0.36, 0.43, 0.46, 0.41]
         ])
    pts[1] = pts[1] + 0.055
    pts = pts.T * 2**OCT_DEPTH
    
    # pts = np.array(  # Letter O
    #     [[0.51, 0.49, 0.89, 0.12, 0.56, 0.61, 0.68, 0.75, 0.80, 0.84, 0.86,
    #       0.86, 0.85, 0.81, 0.74, 0.67, 0.60, 0.53, 0.49, 0.49, 0.42, 0.35,
    #       0.28, 0.23, 0.18, 0.16, 0.14, 0.13, 0.13, 0.13, 0.15, 0.21, 0.27,
    #       0.37, 0.44, 0.50, 0.54, 0.59, 0.65, 0.71, 0.72, 0.75, 0.75, 0.74,
    #       0.72, 0.67, 0.62, 0.54, 0.49, 0.44, 0.37, 0.31, 0.26, 0.23, 0.22,
    #       0.22, 0.23, 0.24, 0.27, 0.30, 0.34, 0.37, 0.42, 0.46, 0.49, 0.54,
    #       0.58, 0.62, 0.68, 0.71, 0.74, 0.76, 0.77, 0.76, 0.75, 0.74, 0.73,
    #       0.64, 0.59, 0.55, 0.44, 0.40, 0.33, 0.29, 0.27, 0.24, 0.23, 0.23,
    #       0.22, 0.22, 0.23, 0.25, 0.28, 0.30, 0.35, 0.38, 0.41, 0.43, 0.45,
    #       0.47, 0.55, 0.60, 0.67, 0.71, 0.70, 0.73, 0.72, 0.71, 0.71, 0.68,
    #       0.66],
    #      [0.92, 0.09, 0.51, 0.50, 0.91, 0.90, 0.85, 0.79, 0.71, 0.60, 0.53,
    #       0.44, 0.37, 0.30, 0.21, 0.15, 0.11, 0.09, 0.09, 0.93, 0.91, 0.90,
    #       0.87, 0.82, 0.76, 0.68, 0.62, 0.54, 0.47, 0.41, 0.33, 0.21, 0.15,
    #       0.10, 0.09, 0.82, 0.81, 0.79, 0.74, 0.66, 0.63, 0.52, 0.48, 0.41,
    #       0.35, 0.28, 0.24, 0.20, 0.19, 0.19, 0.21, 0.24, 0.31, 0.40, 0.45,
    #       0.50, 0.53, 0.59, 0.65, 0.71, 0.75, 0.78, 0.80, 0.82, 0.82, 0.82,
    #       0.80, 0.77, 0.71, 0.66, 0.60, 0.55, 0.51, 0.48, 0.43, 0.38, 0.33,
    #       0.24, 0.21, 0.20, 0.18, 0.20, 0.21, 0.25, 0.28, 0.36, 0.39, 0.47,
    #       0.51, 0.55, 0.59, 0.63, 0.69, 0.71, 0.77, 0.79, 0.81, 0.82, 0.83,
    #       0.84, 0.83, 0.81, 0.75, 0.69, 0.67, 0.34, 0.33, 0.33, 0.29, 0.27,
    #       0.25]]).T * 2**OCT_DEPTH
    
    # pts = np.array(  # Co-centric rings
    #     [[0.48, 0.53, 0.59, 0.62, 0.65, 0.66, 0.66, 0.64, 0.53, 0.46, 0.40,
    #       0.36, 0.35, 0.35, 0.36, 0.38, 0.41, 0.45, 0.51, 0.57, 0.60, 0.62, 
    #       0.50, 0.48, 0.44, 0.41, 0.39, 0.39, 0.41, 0.44, 0.47, 0.53, 0.57,
    #       0.61, 0.62, 0.63, 0.62, 0.58, 0.55, 0.53, 0.44, 0.51, 0.57, 0.62,
    #       0.66, 0.68, 0.69, 0.68, 0.63, 0.55, 0.44, 0.37, 0.33, 0.31, 0.32,
    #       0.35, 0.39, 0.41, 0.47, 0.52, 0.61, 0.72, 0.78, 0.83, 0.86, 0.85,
    #       0.82, 0.73, 0.59, 0.46, 0.32, 0.19, 0.13, 0.12, 0.12, 0.16, 0.26,
    #       0.37, 0.43, 0.54, 0.68, 0.80, 0.85, 0.87, 0.90, 0.89, 0.84, 0.73,
    #       0.60, 0.41, 0.28, 0.17, 0.12, 0.11, 0.11, 0.12, 0.17, 0.26, 0.38,
    #       0.12, 0.09, 0.22, 0.33, 0.52, 0.68, 0.77, 0.90, 0.92, 0.93, 0.92,
    #       0.91, 0.86, 0.71, 0.56, 0.35, 0.22, 0.08, 0.07, 0.08, 0.14, 0.25,
    #       0.41, 0.49, 0.81, 0.16],
    #      [0.64, 0.64, 0.63, 0.61, 0.56, 0.50, 0.45, 0.40, 0.35, 0.35, 0.36,
    #       0.38, 0.42, 0.47, 0.54, 0.59, 0.61, 0.64, 0.35, 0.35, 0.38, 0.39,
    #       0.38, 0.38, 0.39, 0.42, 0.48, 0.51, 0.56, 0.60, 0.61, 0.61, 0.60,
    #       0.57, 0.53, 0.48, 0.44, 0.40, 0.39, 0.39, 0.69, 0.69, 0.67, 0.64,
    #       0.61, 0.55, 0.48, 0.41, 0.35, 0.32, 0.32, 0.35, 0.39, 0.46, 0.54,
    #       0.62, 0.67, 0.68, 0.69, 0.87, 0.85, 0.81, 0.72, 0.60, 0.47, 0.37,
    #       0.29, 0.22, 0.16, 0.15, 0.17, 0.25, 0.37, 0.49, 0.61, 0.74, 0.81,
    #       0.86, 0.90, 0.91, 0.87, 0.79, 0.69, 0.58, 0.46, 0.37, 0.28, 0.20,
    #       0.15, 0.13, 0.16, 0.26, 0.37, 0.48, 0.58, 0.70, 0.80, 0.87, 0.91,
    #       0.32, 0.54, 0.19, 0.88, 0.95, 0.93, 0.86, 0.72, 0.62, 0.50, 0.39,
    #       0.31, 0.24, 0.15, 0.10, 0.09, 0.11, 0.24, 0.42, 0.66, 0.81, 0.91,
    #       0.95, 0.96, 0.19, 0.67]]).T * 2**OCT_DEPTH
      
    pts_quantised = pts.astype(np.int32)  # quantise to quadtree grid
    pts_quantised = np.unique(pts_quantised, axis=0)  # remove duplicates
    
    # Generate z-order curve for pts (swap x and y for correct z-ordering)
    # z_order = zCurve.par_interlace(pts_quantised.tolist(), dims=2)
    z_order = zCurve.par_interlace(np.fliplr(pts_quantised).tolist(), dims=2)
    # Sort pts by z-order
    order_idx = np.argsort(z_order)
    pts_ordered = pts_quantised[order_idx]

    # Offset points so that grid aligns correctly
    pts_quantised = pts_quantised.astype(np.float32) + 0.5
    pts_ordered = pts_ordered.astype(np.float32) + 0.5
    # Scale to [0, 1] range
    pts /= 2**OCT_DEPTH
    pts_quantised /= 2**OCT_DEPTH
    pts_ordered /= 2**OCT_DEPTH

    ## TODO: SCATTER ORIGINAL POINTS BEFORE ROUNDING, THEN ROUND TO GENERATE
    ##       Z-ORDER CURVE, AND COLOUR ALONG THE CURVE FOR WINDOWS

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1)
    # Plot full z-order curve and original points 
    # ax.plot(*pts_fullgrid_ordered.T, color='black', linewidth=2, alpha=0.3)
    x, y = pts_fullgrid_ordered.T
    ax.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], color='black',
              width=BG_LINE_WIDTH, alpha=0.3, zorder=1, **QUIVER_CONFIG)
    ax.scatter(*pts.T, color='black', s=POINT_SIZE, zorder=3)
    # ax.plot(*pts_ordered.T, marker='o', markersize=5, color='black',
    #         linewidth=2, markerfacecolor='white', markeredgecolor='gray')
    # Plot each window with a different color
    color = iter(mpl.colormaps['Set1'].colors)
    for i in range(0, len(pts_ordered), WINDOW_SIZE):
        # ax.plot(*pts_ordered[i:(i + WINDOW_SIZE)].T,
        #         color=next(color), linewidth=3, **MARKER_CONFIG)
        x, y = pts_ordered[i:(i + WINDOW_SIZE)].T
        ax.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], color=next(color),
                  zorder=2, **QUIVER_CONFIG)
    # ax.plot(*pts_ordered.T, color='black', linewidth=2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.grid()
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.tight_layout()


    #---------------------------- POLAR PLOT ----------------------------------#
    # Plot the polar equivalent of the previous figures
        
    # Generate full grid for z-order in polar coords
    # Correct the coordinate ranges for plotting (angle is least significant
    #   bit, but matplotlib expects it first for plotting)
    phi = np.interp(pts_fullgrid_ordered[:, 1], [0, 1], [-np.pi, np.pi])
    rho = pts_fullgrid_ordered[:, 0]
    pts_fullgrid_polar = np.stack([phi, rho], axis=1)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1, polar=True)
    # ax.plot(*pts_fullgrid_polar.T, color='black', linewidth=2)
    phi, rho = pts_fullgrid_polar.T
    ax.quiver(phi[:-1], rho[:-1], phi[1:]-phi[:-1], rho[1:]-rho[:-1],
              width=BG_LINE_WIDTH, color='black', zorder=1, **QUIVER_CONFIG)
    ax.set_aspect('equal', adjustable='box')
    ax.set_rmax(1)
    ax.set_rticks(major_ticks)
    ax.set_xticks(np.pi/180. * np.linspace(180, -180, 2**OCT_DEPTH, endpoint=False))
    ax.set_thetalim(-np.pi, np.pi)
    # ax.set_thetagrids(0123)
    ax.grid(True)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.tight_layout()


    # Convert custom points to polar coords
    pts_polar = pts - 0.5
    phi = np.arctan2(pts_polar[:, 1], pts_polar[:, 0])
    rho = np.sqrt(pts_polar[:, 0]**2 + pts_polar[:, 1]**2) * 2
    rho /= (rho.max() + 0.05)   # rescale rho to fill plot
    pts_polar = np.stack([rho, phi], axis=1)
    pts_polar_plot = np.stack([phi, rho], axis=1)  # matplotlib expects phi first

    # TODO: Scale to [0 ... 2**d] range, quantise, scale back
    # Get z-ordering in polar coords
    pts_quantised_polar = np.zeros_like(pts_polar)
    pts_quantised_polar[:, 0] = np.interp(pts_polar[:, 0], [0, 1],
                                         [0, 2**OCT_DEPTH])
    pts_quantised_polar[:, 1] = np.interp(pts_polar[:, 1], [-np.pi, np.pi],
                                         [0, 2**OCT_DEPTH])
    
    pts_quantised_polar = pts_quantised_polar.astype(np.int32)  # quantise to quadtree grid
    pts_quantised_polar = np.unique(pts_quantised_polar, axis=0)  # remove duplicates
    
    # Generate z-order curve for pts (swap x and y for correct z-ordering)
    # z_order_polar = zCurve.par_interlace(pts_quantised_polar.tolist(), dims=2)
    z_order_polar = zCurve.par_interlace(np.fliplr(pts_quantised_polar).tolist(), dims=2)
    # Sort pts by z-order
    order_idx_polar = np.argsort(z_order_polar)
    pts_ordered_polar = pts_quantised_polar[order_idx_polar]

    # Offset points so that grid aligns correctly
    pts_quantised_polar = pts_quantised_polar.astype(np.float32) + 0.5
    pts_ordered_polar = pts_ordered_polar.astype(np.float32) + 0.5
    
    # Correct the coordinate ranges for plotting (angle is least significant
    #   bit, but matplotlib expects it first for plotting)
    phi = np.interp(pts_ordered_polar[:, 1], [0, 2**OCT_DEPTH], [-np.pi, np.pi])
    rho = np.interp(pts_ordered_polar[:, 0], [0, 2**OCT_DEPTH], [0, 1])
    pts_ordered_polar = np.stack([phi, rho], axis=1)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1, polar=True)
    # Plot full z-order curve and original points 
    # ax.plot(*pts_fullgrid_polar.T, color='black', linewidth=2, alpha=0.3)
    phi, rho = pts_fullgrid_polar.T
    ax.quiver(phi[:-1], rho[:-1], phi[1:]-phi[:-1], rho[1:]-rho[:-1],
              color='black', width=BG_LINE_WIDTH, alpha=0.3, zorder=1,
              **QUIVER_CONFIG)
    ax.scatter(*pts_polar_plot.T, color='black', s=POINT_SIZE, zorder=3)
    # ax.plot(*pts_ordered_polar.T, marker='o', markersize=5, color='black',
    #         linewidth=2, markerfacecolor='white', markeredgecolor='gray')
    # Plot each window with a different color
    color = iter(mpl.colormaps['Set1'].colors)
    for i in range(0, len(pts_ordered_polar), WINDOW_SIZE):
        # ax.plot(*pts_ordered_polar[i:(i + WINDOW_SIZE)].T,
        #         color=next(color), linewidth=3, **MARKER_CONFIG)
        phi, rho = pts_ordered_polar[i:(i + WINDOW_SIZE)].T
        ax.quiver(phi[:-1], rho[:-1], phi[1:]-phi[:-1], rho[1:]-rho[:-1],
                  color=next(color), zorder=2, **QUIVER_CONFIG)
    # ax.plot(*pts_ordered_polar.T, color='black', linewidth=2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_rmax(1)
    ax.set_rticks(major_ticks)
    ax.set_xticks(np.pi/180. * np.linspace(180, -180, 2**OCT_DEPTH, endpoint=False))
    ax.set_thetalim(-np.pi, np.pi)
    # ax.set_thetagrids(0123)
    ax.grid(True)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.tight_layout()
    
    plt.show()