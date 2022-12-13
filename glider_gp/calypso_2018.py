""" Analysis on 2018 Calypso data """

import numpy as np
import os
from pkg_resources import resource_filename

import xarray


from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

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

def fit_spray():#rel_hours, lons_spray, lats_spray, temp_spray):

    # Prep for the GP
    rel_hours, lons_spray, lats_spray, temp_spray = prep_one_spray()

    X_train = np.array([ [t, lon, lat] for t, lon, lat in zip(
        rel_hours, lons_spray, lats_spray)])

    mean_temp = np.mean(temp_spray)
    y_train = temp_spray - mean_temp

    # Kernel
    kRBF_3D = RBF(length_scale=[1,1,1], 
                  length_scale_bounds=[
                      (200, 310), #(1, 300), 
                      (0.01, 1),
                      (0.01, 0.2)]) 
    # Regressor
    gp_3D = GaussianProcessRegressor(
        kernel=kRBF_3D, 
        n_restarts_optimizer=20)
    print("Fitting..")
    gp_3D.fit(X_train, y_train)
    print(f"kernel: {gp_3D.kernel_}")

    t_test = 500
    # Simple test
    y_test, y_std = gp_3D.predict(X_train[0:1], 
                                    return_std=True)
#       [102.18888889,  -1.8489175 ,  36.55766   ],
    # Works but not once I add the time

    # Surface
    chk_surface(lons_spray, lats_spray, temp_spray, gp_3D)
    #chk_surface(lons_spray, lats_spray, temp_spray, gp_3D,
    #            t_test=500)

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
                                    

    embed(header='102 of calypso_2018.py')
    close_t = np.abs(rel_hours - 100) < 20

def chk_surface(lons_spray, lats_spray, temp_spray,
                gp_3D, t_test=100):

    mean_temp = np.mean(temp_spray)

    # Examine
    npts = 100
    lats = np.linspace(np.min(lats_spray),
                           np.max(lats_spray), npts)
    lons = np.linspace(np.min(lons_spray),
                           np.max(lons_spray), npts)
    lon_grid = np.outer(lons, np.ones(lats.size))
    lat_grid = np.outer(np.ones(lons.size), lats)

    X_grid = np.array([ [t_test, lon, lat] for lon, lat in zip(
        lon_grid.flatten(), lat_grid.flatten())])
    y_pred3D, y_std = gp_3D.predict(X_grid, 
                                    return_std=True)
    T_test = y_pred3D.reshape(lon_grid.shape) + mean_temp

    # Plot
    ax = plotting.plot_surface(lon_grid, lat_grid, 
                               T_test, show=False)
    ax.scatter(lons_spray, lats_spray, temp_spray, 
               marker='o', color='k')
    plt.show()


# Command line execution
if __name__ == '__main__':
    fit_spray()