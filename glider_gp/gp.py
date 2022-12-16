import numpy as np

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor


def define_kernel(pargs):
    # Kernel
    kRBF_3D = RBF(length_scale=[1,1,1], 
                  length_scale_bounds=pargs['RBF_bounds'])
    total_kernel = kRBF_3D + WhiteKernel()

    return total_kernel

def explore_L(pair:str, gp_3D, ngrid=30):

    best_theta = gp_3D.kernel_.theta
    if len(best_theta) != 4:
        raise IOError("Expecting an RBF and White kernel")

    # Indices
    idx_dict = dict(time=0, x=1, y=2, sig=3)
    i_idx, j_idx = [idx_dict[item] for item in pair.split(',')]

    i_bounds = gp_3D.kernel_.bounds[i_idx]
    j_bounds = gp_3D.kernel_.bounds[j_idx]
    i_vals = np.linspace(i_bounds[0], i_bounds[1], ngrid)
    j_vals = np.linspace(j_bounds[0], j_bounds[1], ngrid)

    maxL = []
    print(f"Generating maxL grid for pair: {pair}")
    for ival in i_vals:
        for jval in j_vals:
            # Set the noise
            theta = best_theta.copy()
            theta[i_idx] = ival
            theta[j_idx] = jval
            #
            L = gp_3D.log_marginal_likelihood(theta)
            maxL.append(L)

    maxL = np.reshape(maxL, (ngrid,ngrid))
    maxL -= np.max(maxL)

    return maxL, i_vals, j_vals


    #plt.clf()
    #ax = plt.gca()
    #ax.plot(log_noise_val, maxL)
    #plt.show()
    embed(header='204 of calypso_2018.py')
