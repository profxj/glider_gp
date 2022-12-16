import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

def plot_surface(xgrid, ygrid, f, show=True):
    plt.clf()

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')            
    surf = ax.plot_surface(xgrid, ygrid, f,
                        rstride=1, cstride=1, 
                        cmap='jet', linewidth=0, 
                        antialiased=False)

    if show:
        plt.show()

    return ax

def plot_grid_maxL(output_file:str):

    output = np.load(output_file)
    print(output['best_theta'])

    fig = plt.figure(figsize=(10, 5))
    plt.clf()
    gs = gridspec.GridSpec(2,3)

    pairs = ['t_x', 't_y', 't_sig', 'x_y', 'x_sig', 'y_sig']

    window = 0
    for pair in pairs:
        if pair not in output:
            continue

        # Parse
        iv = output[f'{pair}_iv']
        jv = output[f'{pair}_jv']


        ax = plt.subplot(gs[window])
        img = ax.imshow(output[pair].T, origin='lower',
                        extent=[iv[0], iv[-1], jv[0], jv[-1]],
                        aspect='auto', vmin=-10, vmax=0)
        plt.colorbar(img) 
        # Labels

        ax.set_xlabel(pair.split('_')[0])
        ax.set_ylabel(pair.split('_')[1])

        window += 1

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    outfile = output_file.replace('.npz', '.png')
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))