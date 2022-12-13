""" Analysis on 2018 Calypso data """

import numpy as np
import os
from pkg_resources import resource_filename

import xarray


from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

from pypeit.core import fitting

from matplotlib import pyplot as plt

from glider_gp import plotting

from IPython import embed

'''
def load_data():
    """ Load data from .mat file """
    import scipy.io as sio
    data = sio.loadmat('calypso_2018.mat')
    return data
'''

def load_data():
    cfile = os.path.join(resource_filename(
        'glider_gp', 'data'), 'Calypso.nc')
    ds = xarray.open_dataset(cfile)
    return ds

def prep_one_spray(spray=0, field='temperature',
                      depth=50):
    # Load
    ds = load_data()

    # Grab the spray
    profile = ds.trajectory_index == spray

    # Parse the data
    depth_idx = np.argmin(np.abs(ds.depth.data-depth))
    temp_spray = ds[field].data[depth_idx, profile]

    # Cut by region
    if spray == 0:
        lon_cut, lat_cut =  -1.,37.
    elif spray == 1:
        lon_cut, lat_cut =  -0.9, 37.
    cut_reg = (ds.lon[profile] < lon_cut) & (ds.lat[profile] < lat_cut)
    temp_spray = temp_spray[cut_reg]


    # Deal with time
    time = ds.time[profile].data[cut_reg]
    rel_time = time - time[0]
    rel_hours = rel_time.astype(float) / (1e9*3600.)

    # Prep for the GP
    #lon0 = np.min(ds.lon.data[profile][cut_reg])
    #lat0 = np.min(ds.lat.data[profile][cut_reg])

    lons_spray = ds.lon.data[profile][cut_reg]
    lats_spray = ds.lat.data[profile][cut_reg]

    return rel_hours, lons_spray, lats_spray, temp_spray

def fit_sprays(items, bounds=None, linear_fit=None,
               include_noise=True):
    if bounds is None:
        bounds = [(200, 310), (0.01, 1), (0.01, 0.2)]

    # Prep for the GP
    if items is None:
        print("Loading up spray 0")
        rel_hours, lons_spray, lats_spray, temp_spray = prep_one_spray()
    else:
        rel_hours, lons_spray, lats_spray, temp_spray = items

    X_train = np.array([ [t, lon, lat] for t, lon, lat in zip(
        rel_hours, lons_spray, lats_spray)])

    if linear_fit is None:
        mean_temp = np.mean(temp_spray)
        y_train = temp_spray - mean_temp
    else:
        y_fit = linear_fit.eval(lons_spray, x2=lats_spray)
        y_train = temp_spray - y_fit

    # Kernel
    kRBF_3D = RBF(length_scale=[1,1,1], 
                  length_scale_bounds=bounds)

    # Combine kernels
    if include_noise:
        print("Including noise")
        total_kernel = kRBF_3D + WhiteKernel()
    else:
        total_kernel = kRBF_3D 

    # Regressor
    gp_3D = GaussianProcessRegressor(
        kernel=total_kernel, 
        n_restarts_optimizer=20)
    print("Fitting..")
    gp_3D.fit(X_train, y_train)
    print(f"kernel: {gp_3D.kernel_}")

    # Surface
    chk_surface(lons_spray, lats_spray, temp_spray, gp_3D,
                linear_fit=linear_fit)

    '''
    # Another view
    ts = np.linspace(0, rel_hours.max(), 100)
    X_test = np.array( [[t, -1.8489175 ,  36.55766]
                      for t in ts])
    y_test, y_std = gp_3D.predict(X_test,
                                    return_std=True)
    plt.clf()
    ax = plt.gca()
    ax.plot(ts, y_test + mean_temp)
    plt.show()
    '''

    return gp_3D

def generate_grids(lons_spray, lats_spray, npts=100):

    # Examine
    lats = np.linspace(np.min(lats_spray),
                           np.max(lats_spray), npts)
    lons = np.linspace(np.min(lons_spray),
                           np.max(lons_spray), npts)
    lon_grid = np.outer(lons, np.ones(lats.size))
    lat_grid = np.outer(np.ones(lons.size), lats)

    return lon_grid, lat_grid


def chk_surface(lons_spray, lats_spray, temp_spray,
                gp_3D, t_test=500, linear_fit=None):

    mean_temp = np.mean(temp_spray)

    # Grid
    lon_grid, lat_grid = generate_grids(lons_spray, lats_spray)

    X_grid = np.array([ [t_test, lon, lat] for lon, lat in zip(
        lon_grid.flatten(), lat_grid.flatten())])
    y_pred3D, y_std = gp_3D.predict(X_grid, 
                                    return_std=True)
    T_fit = y_pred3D.reshape(lon_grid.shape)

    if linear_fit:
        y_fit = linear_fit.eval(lon_grid.flatten(), 
                                x2=lat_grid.flatten())
        T_fit += y_fit.reshape(lon_grid.shape)
    else:
        T_fit += mean_temp

    # Plot
    ax = plotting.plot_surface(lon_grid, lat_grid, 
                               T_fit, show=False)
    ax.scatter(lons_spray, lats_spray, temp_spray, 
               marker='o', color='k')
    plt.show()

def fit_linear_surface(lons_spray, lats_spray, field_spray,
                       chk=False):
    order = [1,1]
    fit = fitting.robust_fit(lons_spray, field_spray, order, 
                       x2=lats_spray, function='polynomial2d',
                       upper=2., lower=2.)
    print(f"{lons_spray.size-np.sum(fit.gpm)} points were rejected")

    if chk:
        lon_grid, lat_grid = generate_grids(lons_spray, 
                                            lats_spray)
        # Evaluate
        surf = fit.eval(lon_grid.flatten(), x2=lat_grid.flatten())
        surf = surf.reshape(lon_grid.shape)
        #surf = fit.eval(lon_grid, lat_grid)
        # Plot
        ax = plotting.plot_surface(lon_grid, lat_grid, 
                                surf, show=False)
        ax.scatter(lons_spray, lats_spray, field_spray, 
                marker='o', color='k')
        plt.show()

        embed(header='157 of calypso_2018.py')                    

    return fit
    

def fit_one_spray(include_noise=True):
    fit_sprays(None,
        bounds = [(200, 310), (0.01, 1), (0.01, 1)],
        include_noise=include_noise)

def fit_two_sprays(use_linear=False):
    # Spray 0
    rel_hours0, lons_spray0, lats_spray0, temp_spray0 = prep_one_spray()
    # Spray 1
    rel_hours1, lons_spray1, lats_spray1, temp_spray1 = prep_one_spray(spray=1)

    # Concatenate
    rel_hours = np.concatenate([rel_hours0, rel_hours1])
    lons_spray = np.concatenate([lons_spray0, lons_spray1])
    lats_spray = np.concatenate([lats_spray0, lats_spray1])
    temp_spray = np.concatenate([temp_spray0, temp_spray1])

    # Fit linear surface
    if use_linear:
        linear_fit = fit_linear_surface(lons_spray, lats_spray, temp_spray,
                       chk=False)
    else:
        linear_fit = None                    

    #bounds = [(200, 310), (0.1, 1), (0.1, 0.2)]
    #fit_sprays([rel_hours, lons_spray, lats_spray, 
    #            temp_spray], bounds=bounds)

    bounds = [(200, 310), (0.01, 1), (0.01, 1)]
    gp_3D = fit_sprays([rel_hours, lons_spray, lats_spray, 
                temp_spray], bounds=bounds,
                       linear_fit=linear_fit)
    chk_surface(lons_spray, lats_spray, temp_spray, gp_3D, 
                t_test=500, linear_fit=linear_fit)

    embed(header='156 of calypso_2018.py')

# Command line execution
if __name__ == '__main__':

    # One spray
    #fit_one_spray()

    # Try for 2 sprays
    fit_two_sprays(use_linear=False)