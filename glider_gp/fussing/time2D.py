# imports
import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

from matplotlib import pyplot as plt

from glider_gp import plotting

from IPython import embed

def sin_field(x, y, xp=np.pi, yp=np.pi/2):
    f_xy = np.sin(x/xp) * np.sin(y/yp)
    return f_xy

def generate_model():
    # Spatial
    xv = np.linspace(0., 2*np.pi, 100)
    yv = np.linspace(0., 2*np.pi, 100)
    # Grid
    xgrid = np.outer(xv, np.ones_like(yv))
    ygrid = np.outer(np.ones_like(xv), yv)

    # Spatial field
    f_spatial = sin_field(xgrid, ygrid)

    # Time
    ntime = 50
    tval = np.linspace(0., 10., ntime)
    f_t = np.exp(-tval**2/5.)

    f_total = np.zeros((ntime, len(xv), len(yv)))
    for kk in range(ntime):
        f_total[kk, :, :] = f_spatial * f_t[kk]

    return xgrid, ygrid, tval, f_total

def run_gp():
    # Generate model
    xgrid, ygrid, tval, model = generate_model() 

    # Generate training data
    ntrain = 200
    ran_idx = np.random.choice(np.arange(model.size), 
                               size = ntrain, 
                               replace = False)
    t_idx, x_idx, y_idx = np.unravel_index(ran_idx, 
                                           model.shape)
        
    # Training data
    X_train = np.array(
        [ [tval[t_i], xgrid[x_i, y_i], ygrid[x_i, y_i]] for t_i, x_i, y_i in zip(
                             t_idx, x_idx, y_idx)])
    y_train = np.array(
        [ model[t_i, x_i, y_i] for t_i, x_i, y_i in zip(
                             t_idx, x_idx, y_idx)])

    mean_y = np.mean(y_train)
    std_y = np.std(y_train)
    norm_y = (y_train - mean_y) / std_y

    # Kernel
    kRBF_3D = RBF(length_scale=[1,1,1], 
                  length_scale_bounds=[(0.1, 20*np.pi), 
                                       (0.1, 20*np.pi),
                                       (0.1, 20*np.pi)]) 
    # Regressor
    gp_3D = GaussianProcessRegressor(
        kernel=kRBF_3D, 
        n_restarts_optimizer=50)
    print("Fitting..")
    gp_3D.fit(X_train, norm_y)
    print(f"kernel: {gp_3D.kernel_}")

    # Predict
    XY_grid = np.array( [ [x,y] for x,y in zip(xgrid.flatten(), ygrid.flatten())] )
    tXY_grid = []
    for t_i in tval:
        tXY_grid += [ [t_i, x, y] for x,y in XY_grid ]
    

    y_pred3D, y_std = gp_3D.predict(tXY_grid, 
                                        return_std=True)
    model_pred = y_pred3D.reshape(model.shape) * std_y + mean_y
    
    # Examine one slice
    tslice = 20
    # Model
    plotting.plot_surface(xgrid, ygrid, model[tslice, :, :])
    # Fit
    plotting.plot_surface(xgrid, ygrid, model_pred[tslice, :, :])
    # Resid
    plotting.plot_surface(xgrid, ygrid, model[tslice, ...]-
                 model_pred[tslice,...]) 

    embed(header='54')


    
# Command line execution
if __name__ == '__main__':
    run_gp()

    
    