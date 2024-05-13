#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 18:54:08 2024

@author: vwattin
"""

#%% Import packages

from climada.entity import Entity
from climada.util.constants import ENT_DEMO_TODAY, ENT_DEMO_FUTURE, HAZ_DEMO_H5
from climada.hazard import Hazard

import copy
import numpy as np
import matplotlib.pyplot as plt




#%% 

# Start and end year of the simulation
year_current = 2020
year_future = 2050

## Define the hazards
# Current
haz_current = Hazard.from_hdf5(HAZ_DEMO_H5)
# Future
haz_future = copy.deepcopy(haz_current)
haz_future.intensity *= 1
# Interpolation or growth parameters
haz_intpol_param = None # The interpolation curve degree, convex, linear or concave
haz_growth_func = None # If only hazard is given, the growth rate function

# Exposure parameters
# Current
exp_current =  Entity.from_excel(ENT_DEMO_TODAY).exposures
# Future
exp_future = None
exp_future =  copy.deepcopy(exp_current) 
exp_future.gdf.value *= (1+0.02)**(year_future-year_current)
# Interpolation or growth parameters
exp_intpol_param = None # The growth or dev curve of the hazard
exp_growth_func = lambda year: (1+0.02)**(year-year_current)

# Impact function set
imp_fun_set = Entity.from_excel(ENT_DEMO_TODAY).impact_funcs

# Measure set
measure_set = Entity.from_excel(ENT_DEMO_TODAY).measures

# Risk metrics
# This is a function based on the simulated impact matrix


# %% Functions to be used

import numpy as np
import matplotlib.pyplot as plt

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

x_range = [2020, 2050]
y_range = [100, 200]
param = 2
curve_list, curve_dict, curve_invrt, curve_invrt_dict = interpolate_curve(x_range, y_range, param)
plt.plot(range(x_range[0], x_range[1]+1), curve_list, label='Interpolation curve')
plt.plot(range(x_range[0], x_range[1]+1), curve_invrt, label='Inverted interpolation curve')
plt.legend()

# Interpolate the value at a given x and y
x = 2030
interpolate_value = lambda x, x_range, y_range, param: interpolate_curve(x_range, y_range, param)[1][x]
y = interpolate_value(x, x_range, y_range, param)
print(f"The interpolated y value at x={x} is {y}")

#%% Create the exposure per year

exp_param = 2

## Exposure
# Create the exposure per year 
exp_per_year_dict = {}
# If exp_future then interpolate
if exp_future:
    ## Interpolate the exposure
    for year in range(year_current,year_future +1):
        exp_temp = copy.deepcopy(exp_current)
        # Interpolate the exposure value at each exposure point
        for idx in range(len(exp_temp.gdf.value)):
            # Get exposure value at current and future year
            exp_current_val = exp_current.gdf.value[idx]
            exp_future_val = exp_future.gdf.value[idx]
            # Interpolate the value
            exp_temp.gdf.value[idx] = interpolate_value(year, x_range, [exp_current_val, exp_future_val], exp_param)
        # Add the exposure to the dictionary
        exp_per_year_dict[year] = exp_temp
# If not exp_future then use growth function
else:
    for year in range(year_current,year_future +1):
        exp_temp = copy.deepcopy(exp_current)
        exp_temp.gdf.value *= exp_growth_func(year)
        exp_per_year_dict[year] = exp_temp

# Plot the true exposure value and interpolated exposure value for each year for a random exposure point based on exp_per_year_dict
# If exp_future plot the true exposure value and the interpolated exposure value, where the true exposure value is the value at year_current and year_future as clear points
if exp_future:
    exp_point = np.random.randint(0, len(exp_per_year_dict[year_current].gdf.value))
    exp_point_current = exp_current.gdf.value[exp_point]
    exp_point_future = exp_future.gdf.value[exp_point]
    exp_point_interpolated = [exp_per_year_dict[year].gdf.value[exp_point] for year in range(year_current, year_future+1)]
    plt.plot(range(year_current, year_future+1), exp_point_interpolated, label='Interpolated exposure')
    plt.scatter([year_current, year_future], [exp_point_current, exp_point_future], label='True exposure', color='red')
    plt.xlabel('Year')
    plt.ylabel('Exposure value')
    plt.legend()
    plt.title(f'Exposure value at point {exp_point} for each year') 
    plt.show()
else:
    # Simply plot the exposure value for each year
    exp_point = np.random.randint(0, len(exp_per_year_dict[year_current].gdf.value))
    exp_point_interpolated = [exp_per_year_dict[year].gdf.value[exp_point] for year in range(year_current, year_future+1)]
    plt.plot(range(year_current, year_future+1), exp_point_interpolated, label='Exposure')
    plt.xlabel('Year')
    plt.ylabel('Exposure value')
    plt.legend()
    plt.title(f'Exposure value at point {exp_point} for each year')
    plt.show()



#%% Create the hazard per year

#%% Hazard intensity and frequency interpolation
    
haz_int_param = None
haz_freq_param = 2


## If interpolate the intensity derive the scale factor
if haz_int_param:
    haz_int_future_mean = np.mean(haz_future.intensity)
    haz_int_current_mean = np.mean(haz_current.intensity)
    # Store the scale factor for current and future hazard
    scale_factor_current = haz_int_future_mean/haz_int_current_mean
    scale_factor_future = haz_int_current_mean/haz_int_future_mean

    # Plot the interpolated intensity value for each year from 1 to scale_factor_current 
    # include scale_factor_future to 1 
    curve_dict_current = interpolate_curve(x_range, [1, scale_factor_current], haz_int_param)[1]
    curve_dict_future = interpolate_curve(x_range, [scale_factor_future, 1], haz_int_param)[1]
    plt.plot(range(year_current, year_future+1), [curve_dict_current[year] for year in range(year_current, year_future+1)], label='Current hazard')
    plt.plot(range(year_current, year_future+1), [curve_dict_future[year] for year in range(year_current, year_future+1)], label='Future hazard')
    plt.xlabel('Year')
    plt.ylabel('Intensity value')
    plt.legend()
    plt.title('Interpolated intensity value for each year')
    plt.show()


# Create the hazard per year
haz_current_per_year_dict = {}
haz_future_per_year_dict = {}

# If exp_future then interpolate else use growth function
if haz_future:
    for year in range(year_current,year_future +1):
        
        haz_current_temp = copy.deepcopy(haz_current)
        haz_future_temp = copy.deepcopy(haz_future)
        ## Interpolate the intensity value at each hazard point
        if haz_int_param:
            # Interpolate the intensity for the current hazard
            # Scale the intensity by the scale factor
            haz_current_temp.intensity *= interpolate_value(year, x_range, [1, scale_factor_current], haz_int_param)
            haz_future_temp.intensity *= interpolate_value(year, x_range, [scale_factor_future, 1], haz_int_param)
            
        ## Adjust the frequency of the hazard
        # Adjust the frequency for the current hazard
        for idx in range(len(haz_current_temp.frequency)):
            haz_current_temp.frequency[idx] = interpolate_value(year, x_range, [haz_current_temp.frequency[idx], 0.001], haz_freq_param)
        # Adjust the frequency for the future hazard
        for idx in range(len(haz_future_temp.frequency)):
            haz_future_temp.frequency[idx] = interpolate_value(year, x_range, [0.001, haz_future_temp.frequency[idx]], haz_freq_param)

        # Add the hazard to the dictionary
        haz_current_per_year_dict[year] = haz_current_temp
        haz_future_per_year_dict[year] = haz_future_temp

# Plot the max intensity value for each year for a random hazard point based on haz_per_year_dict
# If haz_future plot the true intensity value and the interpolated intensity value, where the true intensity value is the value at year_current and year_future as clear points
if haz_future:
    # Get a random hazard point where intensity is not zero
    haz_point_current = np.random.choice(np.where(haz_current.intensity.toarray()>0)[0])
    haz_current_max_list = []
    # Get max intensity value for each year for a current hazard point
    for year in range(year_current, year_future+1):
        haz_current_max = np.max(haz_current_per_year_dict[year].intensity[haz_point_current])
        haz_current_max_list.append(haz_current_max)
    # Get max intensity value for each year for a future hazard point
    haz_point_future = np.random.choice(np.where(haz_future.intensity.toarray()>0)[0])
    haz_future_max_list = []
    for year in range(year_current, year_future+1):
        haz_future_max = np.max(haz_future_per_year_dict[year].intensity[haz_point_future])
        haz_future_max_list.append(haz_future_max)
    plt.plot(range(year_current, year_future+1), haz_current_max_list, label='Current hazard')
    plt.plot(range(year_current, year_future+1), haz_future_max_list, label='Future hazard')
    plt.xlabel('Year')
    plt.ylabel('Max intensity value')
    plt.legend()
    plt.title('Max intensity value for each year')
    plt.show()




#%% Create the impact per year for current and future hazard and exposure
from climada.engine import ImpactCalc

# Create the impact per year for current and future hazard and exposure
imp_current_per_year_dict = {'no measures': {year : None for year in range(year_current, year_future+1)}}
imp_future_per_year_dict = {'no measures': {year : None for year in range(year_current, year_future+1)}}

# Begin with no measures
for year in range(year_current, year_future+1):
    imp_current_per_year_dict['no measures'][year] = ImpactCalc(exp_per_year_dict[year], imp_fun_set, haz_current_per_year_dict[year] ).impact()
    imp_future_per_year_dict['no measures'][year] = ImpactCalc(exp_per_year_dict[year], imp_fun_set, haz_future_per_year_dict[year]).impact()

# Begin with no measures
for haz_type, measure_dict in measure_set.get_measure().items():
    for meas_name, measure in measure_dict.items():
        print(meas_name)
        # new impact functions
        imp_current_per_year_dict[meas_name] = {year : None for year in range(year_current, year_future+1)}
        imp_future_per_year_dict[meas_name] = {year : None for year in range(year_current, year_future+1)}
        for year in range(year_current, year_future+1):
            new_exp_current, new_impfs_current, new_haz_current = measure.apply(exp_per_year_dict[year], imp_fun_set, haz_current_per_year_dict[year])
            new_exp_future, new_impfs_future, new_haz_future = measure.apply(exp_per_year_dict[year], imp_fun_set, haz_future_per_year_dict[year])
            # Calculate the impact
            imp_current_per_year_dict[meas_name][year] = ImpactCalc(new_exp_current, new_impfs_current, new_haz_current).impact()
            imp_future_per_year_dict[meas_name][year] = ImpactCalc(new_exp_future, new_impfs_future, new_haz_future).impact()


#%% TOod next time

1) Its the hazard events that you sample to a matrix. The number of hazards is the same the the number of impact events
2) Once you have the hazard events you use the same to create the impact sample df for each measure which you store in a dictionary
3) Then you calculate the metrics for each measure which is a function of the measure and the impact sample dict


#%%
        
import climada.util.yearsets as yearsets
import pandas as pd
import seaborn as sns

n_samples = 1000
measure_str = 'no measures'

def calculate_metrics(measure_str, imp_current_per_year_dict, imp_future_per_year_dict, n_samples):
    # Get the years from the dictionaries
    year_current = min(imp_current_per_year_dict[measure_str].keys())
    year_future = max(imp_future_per_year_dict[measure_str].keys())
    return_periods = [100]

    # Initialize DataFrames to store the samples
    samples_current_df = pd.DataFrame()
    samples_future_df = pd.DataFrame()
    samples_total_df = pd.DataFrame()

    # Initialize DataFrames to store the metrics
    metrics_current_df = pd.DataFrame(index=range(year_current, year_future+1), 
                                      columns=['est_mean', 'true_mean'] + [f'est_rp{rp}' for rp in return_periods] + [f'true_rp{rp}' for rp in return_periods])
    metrics_future_df = pd.DataFrame(index=range(year_current, year_future+1), 
                                     columns=['est_mean', 'true_mean'] + [f'est_rp{rp}' for rp in return_periods] + [f'true_rp{rp}' for rp in return_periods])
    metrics_total_df = pd.DataFrame(index=range(year_current, year_future+1), 
                                     columns=['est_mean', 'true_mean'] + [f'est_rp{rp}' for rp in return_periods] + [f'true_rp{rp}' for rp in return_periods])

    # The rest of your code goes here...
    for year in range(year_current, year_future+1):
        # For the current year
        imp_current = imp_current_per_year_dict[measure_str][year]
        lam = np.sum(imp_current.frequency)
        events_per_year = yearsets.sample_from_poisson(n_samples, lam)
        sampling_vect = yearsets.sample_events(events_per_year, imp_current.frequency)
        imp_per_year = yearsets.compute_imp_per_year(imp_current, sampling_vect)
        samples_current_df[year] = imp_per_year

        # For the future year
        imp_future = imp_future_per_year_dict[measure_str][year]
        lam = np.sum(imp_future.frequency)
        events_per_year = yearsets.sample_from_poisson(n_samples, lam)
        sampling_vect = yearsets.sample_events(events_per_year, imp_future.frequency)
        imp_per_year = yearsets.compute_imp_per_year(imp_future, sampling_vect)
        samples_future_df[year] = imp_per_year

        # Total impact
        samples_total_df[year] = samples_current_df[year] + samples_future_df[year]

        # Estimate the metrics
        metrics_current_df.loc[year, 'est_mean'] = np.mean(samples_current_df[year])
        metrics_future_df.loc[year, 'est_mean'] = np.mean(samples_future_df[year])
        metrics_total_df.loc[year, 'est_mean'] = np.mean(samples_total_df[year])
        for rp in return_periods:
            metrics_current_df.loc[year, f'est_rp{rp}'] = np.percentile(samples_current_df[year], 100-100/rp)
            metrics_future_df.loc[year, f'est_rp{rp}'] = np.percentile(samples_future_df[year], 100-100/rp)
            metrics_total_df.loc[year, f'est_rp{rp}'] = np.percentile(samples_total_df[year], 100-100/rp)

        # True metrics
        metrics_current_df.loc[year, 'true_mean'] = imp_current.aai_agg
        metrics_future_df.loc[year, 'true_mean'] = imp_future.aai_agg
        metrics_total_df.loc[year, 'true_mean'] = imp_current.aai_agg + imp_future.aai_agg
        for rp in return_periods:
            metrics_current_df.loc[year, f'true_rp{rp}'] = imp_current.calc_freq_curve([rp]).impact[0]
            metrics_future_df.loc[year, f'true_rp{rp}'] = imp_future.calc_freq_curve([rp]).impact[0]

    return samples_total_df, metrics_current_df, metrics_future_df, metrics_total_df

# Call the function
samples_total_df, metrics_current_df, metrics_future_df, metrics_total_df = calculate_metrics(measure_str, imp_current_per_year_dict, imp_future_per_year_dict, n_samples)


#%%

# Get the common keys
common_keys = set(imp_future_per_year_dict.keys()).intersection(imp_current_per_year_dict.keys())

# Initialize a dictionary to store the total samples DataFrames
samples_total_dict = {}

# Iterate over the common keys
for key in common_keys:
    # Calculate the metrics
    samples_total_df = calculate_metrics(key, imp_current_per_year_dict, imp_future_per_year_dict, n_samples)[0]
    
    # Store the total samples DataFrame in the dictionary
    samples_total_dict[key] = samples_total_df

#%%
    

#%% Plot the sample distribution for each year

import matplotlib.pyplot as plt

# Define color map
colors = plt.get_cmap('tab10')

# Create a figure and a set of subplots
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Plot for current hazard
metrics_current_df['est_mean'].plot(style='--', color=colors(0), ax=ax[0], label='est_mean')
metrics_current_df['true_mean'].plot(style='-', color=colors(0), ax=ax[0], label='true_mean')
for i, rp in enumerate(return_periods):
    metrics_current_df[f'est_rp{rp}'].plot(style='--', color=colors(i+1), ax=ax[0], label=f'est_rp{rp}')
    metrics_current_df[f'true_rp{rp}'].plot(style='-', color=colors(i+1), ax=ax[0], label=f'true_rp{rp}')
ax[0].set_title('Current hazard')
ax[0].set_ylabel('Impact value')
ax[0].set_xlabel('Year')
ax[0].legend()

# Plot for future hazard
metrics_future_df['est_mean'].plot(style='--', color=colors(0), ax=ax[1], label='est_mean')
metrics_future_df['true_mean'].plot(style='-', color=colors(0), ax=ax[1], label='true_mean')
for i, rp in enumerate(return_periods):
    metrics_future_df[f'est_rp{rp}'].plot(style='--', color=colors(i+1), ax=ax[1], label=f'est_rp{rp}')
    metrics_future_df[f'true_rp{rp}'].plot(style='-', color=colors(i+1), ax=ax[1], label=f'true_rp{rp}')
ax[1].set_title('Future hazard')
ax[1].set_ylabel('Impact value')
ax[1].set_xlabel('Year')
ax[1].legend()

plt.tight_layout()
plt.show()

# Plot the total impact no subplots
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
metrics_total_df['est_mean'].plot(style='--', color=colors(0), ax=ax, label='est_mean')
metrics_total_df['true_mean'].plot(style='-', color=colors(0), ax=ax, label='true_mean')
for i, rp in enumerate(return_periods):
    metrics_total_df[f'est_rp{rp}'].plot(style='--', color=colors(i+1), ax=ax, label=f'est_rp{rp}')
ax.set_title('Total impact')
ax.set_ylabel('Impact value')
ax.set_xlabel('Year')
ax.legend()
plt.show()

# %%
