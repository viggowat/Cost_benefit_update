#%% Import packages

from climada.entity import Entity
from climada.util.constants import ENT_DEMO_TODAY, ENT_DEMO_FUTURE, HAZ_DEMO_H5
from climada.hazard import Hazard
from climada.engine import ImpactCalc
import climada.util.yearsets as yearsets


import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#%% Default functions

def exp_expnl_growth(exp_dict, year, growth_rate=0.04):
    year_start = list(exp_dict.keys())[0]
    exp_temp = copy.deepcopy(exp_dict[year_start])
    # Apply the exponential growth rate to the exposure value at each exposure point
    exp_temp.gdf.value = exp_temp.gdf.value * (1+growth_rate)**(year-year_start)
    return exp_temp


def haz_growth(haz_sets_dict, year,year_future, inten_multi=1.5, freq_multi=2, intr_param=1):
    year_start = list(haz_sets_dict.keys())[0]
    haz_temp = copy.deepcopy(haz_sets_dict[year_start])
    # Apply multiply the intensity and frequency by the scale factor given by
    haz_temp.intensity *= interpolate_value(year, [year_start, year_future], [1,inten_multi], intr_param)
    haz_temp.frequency *= interpolate_value(year, [year_start, year_future], [1,freq_multi], intr_param)
    return haz_temp





#%% Utils

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


def get_df_rows(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            return get_df_rows(value)
        elif isinstance(value, pd.DataFrame):
            return len(value)

#%% Exposure interpolation and extrapolation







def generate_exp_per_year(exp_dict, intr_param, year_future, exp_expl_fnc=exp_expnl_growth, plot=False):
    # Get the values of the exposure at a given year
    # Get start exposure and year
    year_start = list(exp_dict.keys())[0]
    exp_start = exp_dict[year_start]

    # Create the exposure per year dictionary 
    exp_per_year_dict = {}

    # Decide if interpolation or extrapolation is needed
    # If more than one exposure year, interpolate the exposure value at each exposure point
    if len(exp_dict.keys()) > 1:

        # Get the future exposure and year
        year_future = list(exp_dict.keys())[1]
        exp_future = exp_dict[year_future]

        # Interpolate the exposure value at each exposure point
        for year in range(year_start,year_future +1):
            exp_temp = copy.deepcopy(exp_start)
            # Interpolate the exposure value at each exposure point
            for idx in range(len(exp_temp.gdf.value)):
                # Get exposure value at start and future year
                exp_start_val = exp_start.gdf.value[idx]
                exp_future_val = exp_future.gdf.value[idx]
                # Interpolate the value
                exp_temp.gdf.value[idx] = interpolate_value(year, [year_start, year_future], [exp_start_val, exp_future_val], intr_param)
            # Add the exposure to the dictionary
            exp_per_year_dict[year] = exp_temp
    # If only one exposure year, extrapolate the exposure value at each exposure point
    else:
        # Extrapolate the exposure value at each exposure point
        for year in range(year_start,year_future +1):
            # Add the exposure to the dictionary
            exp_per_year_dict[year] = exp_expl_fnc(exp_dict, year)

    # Plot the interpolated/extrapolated exposure value at each exposure point
    if plot:
        # Plot random exposure point for each year
        # Get a random exposure point
        exp_point = np.random.randint(0, len(exp_per_year_dict[year_start].gdf.value))
        # Plot the true exposure value at each year use red as scatter color
        years = list(exp_per_year_dict.keys())
        values = [exp_per_year_dict[year].gdf.value[exp_point] for year in years]
        plt.scatter(years, values, color='red', label='Exposure value')
        plt.xlabel('Year')
        plt.ylabel('Exposure value')
        plt.legend()
        plt.title(f'Value at random exposure point (idx= {exp_point}) for each year') 
        plt.show()

    return exp_per_year_dict




#%% Hazard 
# Generate hazard events per year and hazard sets per year

# Check the haz_sets_dict


def generate_haz_sets_per_year(haz_sets_dict, freq_intr_param, year_future, haz_growth_fnc, intens_intr_param=None):
    """

    Parameters
    ----------
    haz_sets_dict : dict
        The dictionary of the hazard sets. Two points
    haz_growth_fnc : function
        The function to extrapolate the hazard intensity at each hazard point.
    freq_intr_param : int
        The frequency interpolation parameter.
    intens_intr_param : int
        The intensity interpolation parameter.
    
    Returns
    -------
    haz_sets_per_year_dict : dict
        The dictionary of the hazard sets per year.

    """ 

    # Get the start hazard and year
    year_start = list(haz_sets_dict.keys())[0]
    haz_start = haz_sets_dict[year_start]

    # Get the hazard type
    haz_type = haz_start.haz_type

    # Create the hazard intensity per interpolation curve
    if len(haz_sets_dict.keys()) > 1:

        # Get the future hazard and year
        year_future = list(haz_sets_dict.keys())[1]
        haz_future = haz_sets_dict[year_future]

        # Check if hazard intensity is to be interpolated
        if intens_intr_param:
            haz_int_future_mean = np.mean(haz_future.intensity)
            haz_int_start_mean = np.mean(haz_start.intensity)
            # Store the scale factor for start and future hazard
            scale_factor_start = haz_int_future_mean/haz_int_start_mean
            scale_factor_future = haz_int_start_mean/haz_int_future_mean

            # Plot the interpolated intensity value for each year from 1 to scale_factor_start 
            # include scale_factor_future to 1 
            curve_dict_start = interpolate_curve([year_start, year_future], [1, scale_factor_start], intens_intr_param)[1]
            curve_dict_future = interpolate_curve([year_start, year_future], [scale_factor_future, 1], intens_intr_param)[1]
            plt.plot(range(year_start, year_future+1), [curve_dict_start[year] for year in range(year_start, year_future+1)], label='start hazard')
            plt.plot(range(year_start, year_future+1), [curve_dict_future[year] for year in range(year_start, year_future+1)], label='Future hazard')
            plt.xlabel('Year')
            plt.ylabel('Intensity value')
            plt.legend()
            plt.title('Interpolated intensity value for each year')
            plt.show() 

        # Create the hazard set per year dictionary
        haz_sets_per_year_dict = {'start': {}, 'future': {}}
        for year in range(year_start,year_future +1):
                
                haz_start_temp = copy.deepcopy(haz_start)
                haz_future_temp = copy.deepcopy(haz_future)

                ## Interpolate the intensity value at each hazard point
                if intens_intr_param:
                    # Interpolate the intensity for the start hazard
                    # Scale the intensity by the scale factor
                    haz_start_temp.intensity *= interpolate_value(year, [year_start, year_future], [1, scale_factor_start], intens_intr_param)
                    haz_future_temp.intensity *= interpolate_value(year, [year_start, year_future], [scale_factor_future, 1], intens_intr_param)
                    
                ## Adjust the frequency of the hazard
                # Adjust the frequency for the start hazard
                for idx in range(len(haz_start_temp.frequency)):
                    haz_start_temp.frequency[idx] = interpolate_value(year, [year_start, year_future], [haz_start_temp.frequency[idx], 0.001], freq_intr_param)
                # Adjust the frequency for the future hazard
                for idx in range(len(haz_future_temp.frequency)):
                    haz_future_temp.frequency[idx] = interpolate_value(year, [year_start, year_future], [0.001, haz_future_temp.frequency[idx]], freq_intr_param)

                # Add the hazard to the dictionary
                haz_sets_per_year_dict['start'][year] = haz_start_temp
                haz_sets_per_year_dict['future'][year] = haz_future_temp

    # Else, if only one hazard year, extrapolate the hazard intensity at each hazard point
    else:
        # Create the hazard set per year dictionary
        haz_sets_per_year_dict = {'start': {}}
        for year in range(year_start,year_future +1):
            haz_sets_per_year_dict['start'][year] = haz_growth_fnc(haz_sets_dict, year)

    # Update the hazard set per year dictionary with the hazard type
    haz_sets_per_year_dict = {haz_type: haz_sets_per_year_dict}

    return haz_sets_per_year_dict




#%% Calculate the impact object per year


def generate_imp_objs_per_year(haz_sets_per_year_dict, exp_per_year_dict, imp_fun_set_per_year, year_start, year_future):
    """
    Calculate the impact object per year

    Parameters
    ----------
    haz_sets_per_year_dict : dict
        The dictionary of the hazard sets per year.
    exp_per_year_dict : dict
        The dictionary of the exposure per year.
    imp_fun_set_per_year : dict
        The dictionary of the impact functions per year.
    year_start : int
        The start year.
    year_future : int
        The future year.

    Returns
    -------
    imp_objs_per_year_dict : dict
        The dictionary of the impact objects per year.
    rel_imp_objs_per_year_dict : dict
        The dictionary of the relative impact objects per year.

    """ 

    rel_imp_objs_per_year_dict = {}
    imp_objs_per_year_dict = {}

    # Create the a dummy exposure object
    exp_dummy = copy.deepcopy(exp_per_year_dict[year_start])
    exp_dummy.gdf.value = 1

    # Iterate over the hazard types and calculate the impact objects per year
    for haz_type, haz_dists in haz_sets_per_year_dict.items():
        # Update the dictionaries
        rel_imp_objs_per_year_dict[haz_type] = {}
        imp_objs_per_year_dict[haz_type] = {}
        # For each distribution, calculate the impact per year
        for dist in haz_dists.keys():
            rel_imp_objs_per_year_dict[haz_type][dist] = {}
            imp_objs_per_year_dict[haz_type][dist] = {}
             # Calculate the impact per year
            for year in range(year_start, year_future+1):
                # The relative impact is calculated using the dummy exposure
                rel_imp_objs_per_year_dict[haz_type][dist][year] = ImpactCalc(exp_dummy, imp_fun_set_per_year[year], haz_sets_per_year_dict[haz_type][dist][year]).impact(save_mat=True)
                # The impact is calculated using the actual exposure
                imp_objs_per_year_dict[haz_type][dist][year] = ImpactCalc(exp_per_year_dict[year], imp_fun_set_per_year[year], haz_sets_per_year_dict[haz_type][dist][year]).impact(save_mat=True)

    return imp_objs_per_year_dict, rel_imp_objs_per_year_dict


def generate_event_ids_per_year(imp_dict, n_samples, year_start, year_end):
    """
    Generate the events per year for each hazard type and distribution

    Parameters
    ----------
    imp_dict : dict
        Dictionary with the relative impact per year.
    n_samples : int
        Number of samples to generate.
    year_start : int
        Start year.
    year_end : int
        End year.

    Returns
    -------
    events_per_year_dict : dict
        Dictionary with the events per year for each hazard type and distribution.

    """
    
    # Create the dictionary for the events per year
    events_per_year_dict = {}

    # Generate the events per year
    for haz_type, has_dists in imp_dict.items():
        
        # Create the dictionary for the hazard type
        events_per_year_dict[haz_type] = {}

        # For each distribution
        for dist in has_dists.keys():

            # Create the dictionary for the distribution
            events_per_year_dict[haz_type][dist] = {}

            # Store the events per year in a dataframe
            temp_events_df = pd.DataFrame()

            # For each year
            for year in range(year_start, year_end+1):

                # Generate the events
                temp_imp = imp_dict[haz_type][dist][year]
                temp_yearly_freq = np.sum(temp_imp.frequency)
                temp_events_per_year = yearsets.sample_from_poisson(n_samples, temp_yearly_freq)
                temp_sampling_vect = yearsets.sample_events(temp_events_per_year, temp_imp.frequency)
                temp_events_df[year] = temp_sampling_vect
            
            # Store the events in the dictionary
            events_per_year_dict[haz_type][dist] = temp_events_df

    return events_per_year_dict


# def generate_imp_objs_per_year(haz_sets_per_year_dict, exp_per_year_dict, imp_fun_set_per_year, measure_set, YEAR_START, YEAR_FUTURE):

#     imp_objs_per_year_dict = {'no measures': {}}

#     # Create the impact object per year dictionary with the measures as keys
#     if 'future' in haz_sets_per_year_dict:
#         for time in haz_sets_per_year_dict.keys():
#             imp_objs_per_year_dict['no measures'][time] = {}
#             # Create the impact object per year dictionary
#             for year in range(YEAR_START, YEAR_FUTURE+1):
#                 # Get measure modified objects
#                 imp_objs_per_year_dict['no measures'][time][year] = ImpactCalc(exp_per_year_dict[year], imp_fun_set_per_year[year], haz_sets_per_year_dict[time][year]).impact()
#     else:
#         # Create the impact object per year dictionary
#         for year in range(YEAR_START, YEAR_FUTURE+1):
#             # Get measure modified objects
#             imp_objs_per_year_dict['no measures'][year] = ImpactCalc(exp_per_year_dict[year], imp_fun_set_per_year[year], haz_sets_per_year_dict[year]).impact()


#     # Calculate the impact per year using measures
#     for haz_type, measure_dict in measure_set.get_measure().items():
#         for meas_name, measure in measure_dict.items():
#             # Add the measure to the measure list
#             imp_objs_per_year_dict[meas_name] = {}
#             # Create the impact object per year dictionary with the measures as keys
#             if 'future' in haz_sets_per_year_dict:
#                 for time in haz_sets_per_year_dict.keys():
#                     imp_objs_per_year_dict[meas_name][time] = {}
#                     for year in range(YEAR_START, YEAR_FUTURE+1):
#                         # Get measure modified objects
#                         new_exp_temp, new_impfs_temp, new_haz_temp = measure.apply(exp_per_year_dict[year], imp_fun_set_per_year[year], haz_sets_per_year_dict[time][year])
#                         imp_objs_per_year_dict[meas_name][time][year] = ImpactCalc(new_exp_temp, new_impfs_temp, new_haz_temp).impact()
#             else:
#                 # Create the impact object per year dictionary
#                 for year in range(YEAR_START, YEAR_FUTURE+1):
#                     # Get measure modified objects
#                     new_exp_temp, new_impfs_temp, new_haz_temp = measure.apply(exp_per_year_dict[year], imp_fun_set_per_year[year], haz_sets_per_year_dict[year])
#                     imp_objs_per_year_dict[meas_name][year] = ImpactCalc(new_exp_temp, new_impfs_temp, new_haz_temp).impact()

#     return imp_objs_per_year_dict


#%% Generate the accumulated relative and absolute impact per year and exposure point given the events ids per year



def generate_imp_per_year(events_ids_per_year_dict, rel_imp_objs_per_year_dict, exp_per_year_dict, YEAR_START, YEAR_FUTURE):
    """
    Generate the accumulated relative and absolute impact per year and exposure point for all hazard types and distributions

    Parameters
    ----------
    events_ids_per_year_dict : dict
        Dictionary with the events ids per year and hazard type and distribution.
    rel_imp_objs_per_year_dict : dict

    exp_per_year_dict : dict
        Dictionary with the exposure per year.
    YEAR_START : int
        Start year.
    YEAR_FUTURE : int
        End year.

    Returns
    -------
    abs_impact_per_year_dict : dict
        Accumulated absolute impact per year for all hazard types and distributions.
    abs_impact_per_exp_year_dict : dict
        Accumulated absolute impact per year and exposure point for all hazard types and distributions.
    -------

    """
    # Get the number of samples and exposure points
    n_samples = get_df_rows(events_ids_per_year_dict) # Number of samples
    nbr_of_exp_points,_ = exp_per_year_dict[YEAR_START].gdf.shape # Number of exposure points

    # Create the total relative impact per year data frame where each cell is an array of zeros with the length of the exposure pionts
    impact_per_exp_year_base_df = pd.DataFrame(index=range(n_samples), columns=range(YEAR_START, YEAR_FUTURE+1))
    impact_per_year_base_df = pd.DataFrame(index=range(n_samples), columns=range(YEAR_START, YEAR_FUTURE+1))
    for year in range(YEAR_START, YEAR_FUTURE+1):
        impact_per_exp_year_base_df[year] = [np.zeros(nbr_of_exp_points) for i in range(n_samples)]
        impact_per_year_base_df[year] = np.zeros(n_samples)

    # Create the accumulated relative impact per simulation containers
    # Per expsonure point and year
    rel_impact_per_exp_year_dict = {}
    abs_impact_per_exp_year_dict = {}
    # Per year
    abs_impact_per_year_dict = {}


    # For each hazard type
    for haz_type in events_ids_per_year_dict.keys():

        # Create the accumulated relative impact per simulation data frames
        # Per expsonure point and year
        rel_impact_per_exp_year_dict[haz_type] = {}
        abs_impact_per_exp_year_dict[haz_type] = {}
        # Per year
        abs_impact_per_year_dict[haz_type] = {}


        # For each distribution
        for dist in events_ids_per_year_dict[haz_type].keys():

            # Create the accumulated impact per year data frames
            # Per expsonure point and year
            rel_impact_per_exp_year_dict[haz_type][dist] = impact_per_exp_year_base_df.copy()
            abs_impact_per_exp_year_dict[haz_type][dist] = impact_per_exp_year_base_df.copy()

            # Get the events ids per year data frame
            temp_event_ids_df = events_ids_per_year_dict[haz_type][dist]

            # Iterate through each cell value
            for year in temp_event_ids_df.columns:
                # Get the impact matrix for the given year
                temp_rel_imp_matrix = rel_imp_objs_per_year_dict[haz_type][dist][year].imp_mat.toarray() # Convert to array for simlicity
                
                # Iterate through each sample
                for sample_id in range(len(temp_event_ids_df)):
                    idx_Events = temp_event_ids_df.loc[sample_id, year]
                    # Per exposure point and year
                    rel_impact_per_exp_year_dict[haz_type][dist].loc[sample_id, year] = np.sum(temp_rel_imp_matrix[idx_Events], axis=0) # -1 because the column index starts at 1
                    abs_impact_per_exp_year_dict[haz_type][dist].loc[sample_id, year] = rel_impact_per_exp_year_dict[haz_type][dist].loc[sample_id, year] * exp_per_year_dict[year].gdf.value.to_numpy()
            
        # Add a total key that sums all the distributions contributions
        rel_impact_per_exp_year_dict[haz_type]['total'] = impact_per_exp_year_base_df.copy()
        abs_impact_per_exp_year_dict[haz_type]['total'] = impact_per_exp_year_base_df.copy()
        # Accumulate the impact per year and exposure point
        for dist in rel_impact_per_exp_year_dict[haz_type].keys():
            if dist != 'total':
                # Iterate through each cell value
                for year in rel_impact_per_exp_year_dict[haz_type]['total'].columns:

                    # Per expsonure point and year
                    # Relative impact
                    rel_impact_per_exp_year_dict[haz_type]['total'][year] += rel_impact_per_exp_year_dict[haz_type][dist][year]
                    rel_impact_per_exp_year_dict[haz_type]['total'][year] = rel_impact_per_exp_year_dict[haz_type]['total'][year].apply(lambda arr: np.minimum(arr, 1))
                    # Absolute impact
                    abs_impact_per_exp_year_dict[haz_type]['total'][year] = rel_impact_per_exp_year_dict[haz_type]['total'][year].apply(lambda x: np.multiply(x, exp_per_year_dict[year].gdf.value.to_numpy()))
                    

    # Aggregate the impact per year and exposure point for all hazard types
    rel_impact_per_exp_year_dict['MULTI'] = {'total': impact_per_exp_year_base_df.copy()}
    abs_impact_per_exp_year_dict['MULTI'] = {'total': impact_per_exp_year_base_df.copy()}

    # Accumulate the impact per year and exposure point
    for haz_type in rel_impact_per_exp_year_dict.keys():
        if haz_type != 'MULTI':
            # Iterate through each cell value
            for year in rel_impact_per_exp_year_dict['MULTI']['total'].columns:
                # Per expsonure point and year but upper bounded by 1
                rel_impact_per_exp_year_dict['MULTI']['total'][year] += rel_impact_per_exp_year_dict[haz_type]['total'][year]
                rel_impact_per_exp_year_dict['MULTI']['total'][year] = rel_impact_per_exp_year_dict['MULTI']['total'][year].apply(lambda arr: np.minimum(arr, 1))
                # Absolute impact
                abs_impact_per_exp_year_dict['MULTI']['total'][year] = rel_impact_per_exp_year_dict['MULTI']['total'][year].apply(lambda x: np.multiply(x, exp_per_year_dict[year].gdf.value.to_numpy()))


        # Aggregate the impact per year for all hazard types
        abs_impact_per_year_dict[haz_type] = abs_impact_per_exp_year_dict[haz_type]['total'].map(np.sum)

    return abs_impact_per_year_dict, abs_impact_per_exp_year_dict


def generate_imp_per_year_using_yearsets(events_ids_per_year_dict, imp_objs_per_year_dict, YEAR_START, YEAR_FUTURE):
    """
    Generate the accumulated absolute impact per year using the yearsets module.
    Good for comparing the results with the previous function.

    Parameters
    ----------
    events_ids_per_year_dict : dict
        Dictionary with the events ids per year and hazard type and distribution.
    imp_objs_per_year_dict : dict
        Dictionary with the impact objects per year.
    YEAR_START : int
        Start year.
    YEAR_FUTURE : int
        End year.

    Returns
    -------
    abs_impact_per_year_dict : dict
        Accumulated absolute impact per year for all hazard types and distributions.
    
    """

    # Get the number of samples and exposure points
    n_samples = get_df_rows(events_ids_per_year_dict) # Number of samples

    # Create the total relative impact per year data frame where each cell is an array of zeros with the length of the exposure pionts
    impact_per_year_base_df = pd.DataFrame(index=range(n_samples), columns=range(YEAR_START, YEAR_FUTURE+1))
    for year in range(YEAR_START, YEAR_FUTURE+1):
        impact_per_year_base_df[year] = np.zeros(n_samples)


    # Per year
    abs_impact_per_dist_year_dict = {}


    # For each hazard type
    for haz_type in events_ids_per_year_dict.keys():
        abs_impact_per_dist_year_dict[haz_type] = {}
        for dist in events_ids_per_year_dict[haz_type].keys():
            abs_impact_per_dist_year_dict[haz_type][dist] = {}
            
            temp_df = events_ids_per_year_dict[haz_type][dist]

            # make a zero data frame with same numer of rows and columns as the temp_df
            abs_impact_per_dist_year_dict[haz_type][dist] = impact_per_year_base_df.copy()

            # for each row and column
            for idx_row in range(temp_df.shape[0]):
                for year in temp_df.columns:
                    # get the event ids for the year
                    sampling_vect = list(np.array([temp_df.loc[idx_row, year]]))
                    imp = imp_objs_per_year_dict[haz_type][dist][year]
                    abs_impact_per_dist_year_dict[haz_type][dist].loc[idx_row, year] = yearsets.compute_imp_per_year(imp, sampling_vect)
        
        # Accumulate the impact per year and exposure point
        abs_impact_per_dist_year_dict[haz_type]['total'] = impact_per_year_base_df.copy()
        for dist in events_ids_per_year_dict[haz_type].keys():
            # Iterate through each cell value
            for year in abs_impact_per_dist_year_dict[haz_type]['total'].columns:
                abs_impact_per_dist_year_dict[haz_type]['total'][year] += abs_impact_per_dist_year_dict[haz_type][dist][year]

        # Aggregate the impact per year for all hazard types
        
    # Aggregate the impact per year for all hazard types
    abs_impact_per_year_dict = {haz_type: abs_impact_per_dist_year_dict[haz_type]['total'].copy() for haz_type in abs_impact_per_dist_year_dict.keys()}
    abs_impact_per_year_dict['MULTI'] = impact_per_year_base_df.copy()

    # Accumulate the impact per year and exposure point
    for haz_type in events_ids_per_year_dict.keys():
        if haz_type != 'MULTI':
            abs_impact_per_year_dict['MULTI'] += abs_impact_per_year_dict[haz_type]

    return abs_impact_per_year_dict