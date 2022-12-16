import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from glider_gp import plotting


def figs_2018():
    # 2018

    # max L
    #plotting.plot_grid_maxL('output/calypso_2018_d50T.npz')
    #plotting.plot_grid_maxL('output/calypso_2018_d100T.npz')
    #plotting.plot_grid_maxL('output/calypso_2018_d150T.npz')
    #plotting.plot_grid_maxL('output/calypso_2018_d200T.npz')

    # best
    best = dict(t=[], x=[], y=[], sig=[])
    depths = [50, 100, 150, 200]
    for depth in depths:
        output = np.load(
            f'output/calypso_2018_d{depth}T.npz')
        # Grab
        best['t'].append(output['best_theta'][0])
        best['x'].append(output['best_theta'][1])
        best['y'].append(output['best_theta'][2])
        best['sig'].append(output['best_theta'][3])

    # Plot
    fig = plt.figure(figsize=(10, 10))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    for hyper, kk in zip(best.keys(), range(4)):
        ax = plt.subplot(gs[kk])

        ax.plot(depths, np.exp(best[hyper]))

        ax.set_xlabel('depth (m)')
        ax.set_ylabel(hyper)

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    #outfile = output_file.replace('.npz', '.png')
    outfile = 'output/calypso_2018_best.png'
    plt.savefig(outfile, dpi=300)
    #plt.show()
    print('Wrote {:s}'.format(outfile))

def figs_2019():
    #plotting.plot_grid_maxL('output/calypso_2019_d50T.npz')
    plotting.plot_grid_maxL('output/calypso_2019_d200T.npz')

if __name__ == '__main__':

    # 2018
    #figs_2018()

    # 2019
    figs_2019()