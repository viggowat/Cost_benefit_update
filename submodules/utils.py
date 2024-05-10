


#%% Importing necessary libraries
import numpy as np



#%% Functions for interpolation

def interpolate_curve(x_range, y_range=[0,1], param=0):
    """
    Interpolate a curve between two points

    Parameters
    ----------
    x_range : list
        The start and end year of the curve.
    y_range : list, optional
        The start and end value of the curve. The default is [0,1].
    param : int, optional
        The degree of the curve. The default is 0.

    Returns
    -------
    curve_list : list
        The list of the interpolated values.
    curve_dict : dict
        The dictionary of the interpolated values.
    curve_invrt : list
        The inverted list of the interpolated values.
    curve_invrt_dict : dict
        The inverted dictionary of the interpolated values.

    """ 
    x_diff = x_range[1] - x_range[0] + 1
    curve_list = np.linspace(0, 1, x_diff)**param
    
    if y_range[0] == 0:
        curve_list *= y_range[1]
    else:
        curve_list *= y_range[1] - y_range[0]
        curve_list += y_range[0]
    
    curve_dict = {year: curve_list[idx] for idx, year in enumerate(range(x_range[0], x_range[1]+1))}
    
    curve_invrt = curve_list[::-1]
    curve_invrt_dict = {year: curve_invrt[idx] for idx, year in enumerate(range(x_range[0], x_range[1]+1))}
                          
    return curve_list, curve_dict, curve_invrt, curve_invrt_dict


# Create a function to interpolate the value of the curve at a given year
def interpolate_value(x, x_range, y_range, param):
    """
    Interpolate the value of the curve at a given year

    Parameters
    ----------
    x : int
        The year at which the value is to be interpolated.
    x_range : list
        The start and end year of the curve.
    y_range : list
        The start and end value of the curve.
    param : int
        The degree of the curve.

    Returns
    -------
    float
        The interpolated value of the curve at the given year.

    """
    return interpolate_curve(x_range, y_range, param)[1][x]