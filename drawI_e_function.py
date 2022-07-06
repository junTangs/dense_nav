import numpy as np
import matplotlib.pyplot as plt


def gaussian(x,sigma):
    return np.exp(-x**2/(2*sigma**2))

def gen_2d_grid(x_range,y_range,resolution):
    """
    Generate a 2D grid of points
    :param x_range: range of x values
    :param y_range: range of y values
    :param resolution: resolution of the grid
    :return: a 2D grid of points
    """
    x_grid = np.arange(x_range[0],x_range[1],resolution)
    y_grid = np.arange(y_range[0],y_range[1],resolution)
    x_grid,y_grid = np.meshgrid(x_grid,y_grid)
    return x_grid,y_grid


def distance(x_grid,y_grid,x0,y0):
    """
    Calculate the distance between a point and a grid
    :param x_grid: x values of the grid
    :param y_grid: y values of the grid
    :param x0: x value of the point
    :param y0: y value of the point
    :return: a 2D grid of distances
    """
    return np.sqrt((x_grid-x0)**2+(y_grid-y0)**2)


def att(x_grid,y_grid,x0,y0):
    """
    Calculate the attention field of a point
    :param x_grid: x values of the grid
    :param y_grid: y values of the grid
    :param x0: x value of the point
    :param y0: y value of the point
    :return: a 2D grid of attention values
    """
    
    return 0.5*distance(x_grid,y_grid,x0,y0)**2



def rep(x_grid,y_grid,x0,y0):
    """
    Calculate the repulsion field of a point
    :param x_grid: x values of the grid
    :param y_grid: y values of the grid
    :param x0: x value of the point
    :param y0: y value of the point
    :return: a 2D grid of repulsion values
    """
    dist = distance(x_grid,y_grid,x0,y0)
    return gaussian(dist,0.5)



def cal_apf(goal,humans,x_range,y_range,resolution):
    x_grid , y_grid = gen_2d_grid(x_range,y_range,resolution)
    a = att(x_grid,y_grid,goal[0],goal[1])
    r = np.zeros_like(a)
    for human in humans:
        r += rep(x_grid,y_grid,human[0],human[1])
    t = a + r
    return t,a,r




def draw_apf(goal,humans,x_range,y_range,resolution,title="",save_path=None):
    t,a,r = cal_apf(goal,humans,x_range,y_range,resolution)
    fig = plt.figure()
    plt.title(title)
    plt.subplot(1,3,1)
    plt.imshow(t,cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(a,cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(r,cmap='gray')
    if save_path is not None:
        plt.savefig(save_path,dpi=600)
        
    