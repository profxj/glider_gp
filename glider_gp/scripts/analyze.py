""" Script to grab LLC model data """

from IPython import embed

'''
LLC docs: https://mitgcm.readthedocs.io/en/latest/index.html
'''

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Analyze GP data')
    parser.add_argument("inp_file", type=str, help="Inptut file [JSON]")
    #parser.add_argument("--model", type=str, default='LLC4320',
    #                    help="LLC Model name.  Allowed options are [LLC4320]")
    #parser.add_argument("--var", type=str, default='Theta',
    #                    help="LLC data variable name.  Allowed options are [Theta, U, V, Salt]")
    #parser.add_argument("--istart", type=int, default=0,
    #                    help="Index of model to start with")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def main(args):
    """ Run
    """
    import numpy as np
    import os

    from matplotlib import pyplot as plt

    from sklearn.gaussian_process import GaussianProcessRegressor

    from glider_gp import utils as glider_utils
    from glider_gp import calypso
    from glider_gp import gp

    # Load input file
    pargs = glider_utils.loadjson(args.inp_file)

    # Load the data
    if pargs['dataset'] == 'calypso':
        data = calypso.load_for_gp(pargs)

    # Prep for GP
    X_train = np.array([ [t, lon, lat] for t, lon, lat in zip(
        data['time'], data['lons'], data['lats'])])

    if 'linear_fit' in pargs['preproc']:
        raise IOError("linear_fit not implemented yet")
        #y_fit = linear_fit.eval(lons_spray, x2=lats_spray)
        y_train = data['field'] - y_fit
    else:
        mean_temp = np.mean(data['field'])
        y_train = data['field'] - mean_temp

    # Kernel
    total_kernel = gp.define_kernel(pargs)

    # Regressor
    gp_3D = GaussianProcessRegressor(
        kernel=total_kernel, 
        n_restarts_optimizer=20)

    output = {}

    # Fit?
    if pargs['fit']:
        print("Fitting..")
        gp_3D.fit(X_train, y_train)
        output['best_theta'] = gp_3D.kernel_.theta
    else:
        raise IOError("Need to have the best fit values already")

    # maxL
    if pargs['maxL'] == 'all':
        pairs = ['t_x', 't_y', 't_sig', 'x_y', 'x_sig', 'y_sig']
    elif pargs['maxL'] == 'none':
        pairs = []
    else:
        pairs = pargs['maxL']

    for pair in pairs:
        L, iv, jv = gp.explore_L(pair, gp_3D)
        output[pair] = L
        output[f'{pair}_iv'] = iv
        output[f'{pair}_jv'] = jv

    # Output
    outfile = os.path.join(
        pargs['outdir'], 
        os.path.basename(
            args.inp_file).replace('.json', '.npz'))
    np.savez(outfile, **output)

    '''
    # Plot
    plt.clf()
    ax = plt.gca()
    img = ax.imshow(maxL[pair].T, origin='lower',
                    extent=[iv[0], iv[-1], jv[0], jv[-1]],
                    aspect='auto', vmin=-10, vmax=0)
    plt.colorbar(img) 
    plt.show()
    '''

    embed(header='44 of analyze.py')