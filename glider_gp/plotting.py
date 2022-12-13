from matplotlib import pyplot as plt

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