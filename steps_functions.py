# Import the necessary libraries
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from climada.engine import ImpactCalc

import functions as fcn


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




#%% Functions related to the exposure objects

# Define the function generate_exp_per_year that generates interpolated and extrapolated exposure sets per year
def generate_exp_sets(exp_dict, intr_param,  future_year=None, growth_rate=0.02):
    """
    Generate interpolated and extrapolated exposure sets per year
    """

    start_year = min(exp_dict.keys())

    #%%% Generate the Exposure given dictionary and update so all the exposure sets have the same geo locations
    # Check if the future_year is None
    if future_year is None:
        future_year = max(exp_dict.keys())


    # Make all the items in the dictionary as list
    for key, itm in exp_dict.items():
        if not isinstance(itm, list):
            exp_dict[key] = [itm]

    # Store all the available exposure sets in a dictionary
    exp_given_dict = {} 

    # Check all the possible exposure value_unit and store them in a list
    exp_value_units = []

    # Check all the possible exposure value_unit
    for year, exp_list in exp_dict.items():
        # Check if the column value_unit exists
        for exp in exp_list:
            # If the column value_unit does not exist, raise an error
            if 'value_unit' not in exp.gdf.columns:
                raise ValueError('The column value_unit should exist in the exposure dataframe.')
            # If the column value_unit exists, ...
            else:
                value_unit = exp.gdf.value_unit.unique()
                # Check if the value_unit column has only one unique value
                if len(value_unit) > 1:
                    raise ValueError('The value_unit column should have only one unique value')
                # Check if the value_unit is not already in the list
                elif value_unit[0] not in exp_value_units:
                    # Store the value_unit in the list
                    exp_value_units.append(value_unit[0])
                    # Create a new key in the exp_given_dict dictionary
                    exp_given_dict[value_unit[0]] = {}
                # Store the exposure in the dictionary but check if the year is already in the dictionary
                if year in exp_given_dict[value_unit[0]]:
                    raise ValueError('The year already exists in the exposure set')
                # Store the exposure in the dictionary
                exp_given_dict[value_unit[0]][year] = exp

    # Check that for the first year the number of exposure objects is the same
    for value_unit, exp_dict in exp_given_dict.items():
        # Check if the number of exposure objects is the same
        if not start_year in exp_dict.keys():
            raise ValueError('Both exposure objects should exist for the first year')
                

    # For each value_unit get the unique exposure geo locations
    exp_geo_locs = {}
    for value_unit, exp_dict in exp_given_dict.items():
        exp_geo_locs[value_unit] = None
        idx = 0
        boo_all_same = True
        # Get the unique geo locations
        for year, exp in exp_dict.items():
            # Get the unique geo locations
            if idx == 0:
                exp_geo_locs[value_unit] = exp.gdf[['longitude', 'latitude']].drop_duplicates()
                idx += 1
            else:
                # Check if the geo locations are the same
                if not exp_geo_locs[value_unit].equals(exp.gdf[['longitude', 'latitude']].drop_duplicates()):
                    print(f'For {value_unit}, the geo locations are different in the exposure sets')
                    boo_all_same = False
                    # Add the additional geo locations
                    exp_geo_locs[value_unit] = pd.concat([exp_geo_locs[value_unit], exp.gdf[['longitude', 'latitude']].drop_duplicates()], ignore_index=True)
                    # Drop duplicates
                    exp_geo_locs[value_unit] = exp_geo_locs[value_unit].drop_duplicates()
                    # Reset index
                    exp_geo_locs[value_unit].reset_index(drop=True, inplace=True)

        # Print total number of geo locations for value_unit
        print(f'Total number of geo locations for {value_unit} is {len(exp_geo_locs[value_unit])}')

        # If the geo locations are not the same add a new exposure point to the ones missing with value 0
        if not boo_all_same:
            for year, exp in exp_dict.items():
            # Print the row exp_geo_locs that does not exist in the exposure set
                # Get the longitude and latitude that do not exist in the exposure set
                merged_df = pd.merge(exp_geo_locs[value_unit], exp.gdf[['longitude', 'latitude']].drop_duplicates(), on=['longitude', 'latitude'], how='outer', indicator=True)
                missing_geo_locs = merged_df[merged_df['_merge'] == 'left_only'][['longitude', 'latitude']]
                #print(f'The missing geo locations in the year {year} are:')
                #print(missing_geo_locs)
                if len(missing_geo_locs) > 0:
                    # Add the missing geo locations rows with value 0 and store them in the exposure set and longitude and latitude columns and other values the same as the first row
                    for idx, row in missing_geo_locs.iterrows():
                        # Get the first row of the exposure set
                        first_row = exp.gdf.iloc[0]
                        # Create a new row with the missing geo location
                        new_row = first_row.copy()
                        new_row['longitude'] = row['longitude']
                        new_row['latitude'] = row['latitude']
                        new_row['value'] = 0
                        # Add the new row to the exposure set without using append
                        exp.gdf.loc[len(exp.gdf)] = new_row
            # check so that the number of rows in the exposure set is the same as the number of geo locations and print the year
            if len(exp.gdf) != len(exp_geo_locs[value_unit]):
                raise ValueError(f'The number of rows in the exposure set is not the same as the number of geo locations in the year {year}')
            if len(exp.gdf) != len(exp_geo_locs[value_unit]):
                print(f'The number of rows in the exposure set is not the same as the number of geo locations in the year {year}')


    #%%% Generate the interppolated Exposure dictionary
                
    # Get the year range for each value_unit
    exp_inter_pol_years = {}
    # Count the number of exposure objects for each value_unit
    for value_unit, exp_dict in exp_given_dict.items():
        exp_inter_pol_years[value_unit] = None
        print(f'The number of exposure objects for {value_unit} is {len(exp_dict)}')
        # Get the year range for each value_unit
        exp_inter_pol_years[value_unit] = [min(exp_dict.keys()), max(exp_dict.keys())]

    # Store all the interpolated and extrapolated exposure sets
    exp_avail_dict = {} # Store all the interpolated and extrapolated exposure sets 

    # Interpolate the exposure sets
    for value_unit in exp_value_units:
        exp_avail_dict[value_unit] = {}
        # Check if the value_unit has more than one exposure set
        if not exp_inter_pol_years[value_unit]:
            exp_avail_dict[value_unit] = exp_given_dict[value_unit]
        else:
            exp_dict = exp_given_dict[value_unit]
            # Interpolate
            for year_start, year_end in zip(list(exp_dict.keys())[:-1], list(exp_dict.keys())[1:]):
                #print(f'The pair of years is {year_start} and {year_end}')
                # Make a subset of the exposure dictionary
                exp_dict_subset = {}
                exp_dict_subset[year_start] = exp_dict[year_start]
                exp_dict_subset[year_end] = exp_dict[year_end]
                # Interpolation parameter
                exp_temp_dict = fcn.generate_exp_per_year(exp_dict_subset, intr_param, future_year)
                # Add the interpolated exposure to the dictionary
                exp_avail_dict[value_unit].update(exp_temp_dict)
            # Extrapolate
            #if future_year not in exp_avail_dict[value_unit]:
            #    # Take last year as the last year in the dictionary
            #    exp_dict_subset = {year_end: exp_dict_subset[year_end]}
            #    exp_temp_dict = fcn.generate_exp_per_year(exp_dict_subset, intr_param, exp_expl_fnc, future_year)
            #    exp_avail_dict[value_unit].update(exp_temp_dict)

    #%%% Generate the scale Exposure dataframe dictionary

    # Make a dictionary where you store the scaleing factor for each year in a data frame
    exp_multipl_dict = {}
    # Calculate the scaling factor for each year
    for value_unit, exp_dict in exp_avail_dict.items():
        # Make a diagonal data frame with zeros and ones in the diagonal
        exp_multipl_dict[value_unit] = pd.DataFrame(index=range(start_year, future_year+1), columns=exp_dict.keys(), dtype=float)
        for year_row in exp_multipl_dict[value_unit].index:
            for year_col in exp_multipl_dict[value_unit].columns:
                if year_row == year_col:
                    exp_multipl_dict[value_unit].loc[year_row, year_col] = 1.0
                else:
                    exp_multipl_dict[value_unit].loc[year_row, year_col] = 0.0

        # Get the max column year
        max_col_year = max(exp_dict.keys())
        # If the max column year is not the last year, extrapolate the values from one and add them to the last year
        if max_col_year != future_year:
            # Extrapolate the exposure value at each exposure point
            for year_row in range(max_col_year,future_year+1):
                # Add the exposure to the dictionary
                exp_multipl_dict[value_unit].loc[year_row, max_col_year] = (1 + growth_rate)**(year_row - max_col_year)

    #%%% Plot the interpolated/extrapolated exposure value at a given or random exposure point
                
    # Plot the interpolated/extrapolated exposure value at each exposure point      
    for value_unit, exp_dict in exp_avail_dict.items():
        # Plot the interpolated/extrapolated exposure value at each exposure point
        # Plot random exposure point for each year
        # Get a random exposure point from first year
        first_year = list(exp_dict.keys())[0]
        exp_point = np.random.randint(0, len(exp_dict[first_year].gdf))
        # Plot the true exposure value at each year use red as scatter color
        years = list(exp_dict.keys())
        values = [exp_dict[year].gdf.value[exp_point] for year in years]
        # Calulate the extrapolated values from exp_multipl_dict[value_unit]
        extra_years = [year for year in  range(max(years)+1, future_year +1)]
        extra_values = [multi*values[-1] for multi in exp_multipl_dict[value_unit].loc[extra_years, max(years)]]
        # Add the yeasr and values
        years += extra_years
        values += extra_values
        plt.scatter(years, values, color='red', label='Exposure value')
        # Get the available exposure years
        given_years = list(exp_given_dict[value_unit].keys())
        given_values = [exp_given_dict[value_unit][year].gdf.value[exp_point] for year in given_years]
        plt.scatter(given_years, given_values, color='blue', label='Given exposure value')
        plt.xlabel('Year')
        plt.ylabel('Exposure value')
        plt.legend()
        plt.title(f' For value_unit {value_unit} - Value at random exposure point (idx= {exp_point}) for each year') 
        plt.show()

    return exp_avail_dict, exp_given_dict, exp_multipl_dict, exp_inter_pol_years







#%% Generate the hazard per year attributes

def generate_haz_sets(haz_dict, intr_param=1, future_year=None):


    # Get the start year
    start_year = min(haz_dict.keys())


    # Check if the future_year is None
    if future_year is None:
        future_year = max(haz_dict.keys())


    #%% Gnereate the haz_given_dict

    # Make all the items in the dictionary as list
    for key, itm in haz_dict.items():
        if not isinstance(itm, list):
            haz_dict[key] = [itm]


    # Check all the possible hazard types and store them in a list
    haz_types_list = []

    # Store all the available hazard sets in a dictionary
    haz_given_dict = {}

    # Check all the possible hazard value_unit
    for year, haz_list in haz_dict.items():
        # Check if the column value_unit exists
        for haz in haz_list:
            if haz.haz_type not in haz_types_list:
                haz_types_list += [haz.haz_type]
                # Store all the available hazard sets in a dictionary
                haz_given_dict[haz.haz_type] = {}
            # Store the hazard in the dictionary but check if the year is already in the dictionary
            haz_given_dict[haz.haz_type][year] = haz


    #%% Gnereate the hazard scaling dataframe dictionary
    haz_param_dict = {}

    # Create the data frame with the scaling factors
    for haz_type, haz_dict in haz_given_dict.items():
        given_years = list(haz_dict.keys())
        # Create the scaling factor dataframe filled with zeros of float type
        haz_scale_df = pd.DataFrame(index=range(start_year, future_year+1), columns=given_years)
        haz_scale_df = haz_scale_df.fillna(0.0)
        # Checck if the given years are only one
        if len(given_years) == 1:
            # Set the scaling factor to 1 for all the years
            haz_scale_df[given_years[0]] = 1
        else:
            # Walk through all pairs of years in given_years
            for i in range(len(given_years)-1):
                # Get the years
                year_0 = given_years[i] # Start year
                year_1 = given_years[i+1] # End year
                # Get the scaling factor for the years
                for idx_year in range(year_0, year_1+1):
                    # Get the scaling factor
                    scaling_factor = interpolate_value(idx_year, [year_0, year_1], [0,1], intr_param)
                    haz_scale_df.loc[idx_year, year_0] = 1- scaling_factor
                    haz_scale_df.loc[idx_year, year_1] = scaling_factor
            # Set the scaling factor to 1 for the last year when the index year is equal or greater than the last given year
            haz_scale_df.loc[haz_scale_df.index >= year_1, year_1] = 1

        # Store the scaling factor dataframe in the dictionary
        haz_param_dict[haz_type] = haz_scale_df
                    


    #%% Plot the scaling factor

    for haz_type, haz_scale_df in haz_param_dict.items():
        # Create the plot
        ax = haz_scale_df.plot(title=haz_type, grid=True, style='-', fontsize=12)
        # Create a DataFrame that only contains the points where the value is one
        haz_scale_df_one = haz_scale_df[haz_scale_df == 1]
        # Plot these points
        ax.plot(haz_scale_df_one, 'o')
        # Add a horizontal line at y=1
        ax.axhline(y=1, color='r', linestyle='--')
        # set the x-axis label
        ax.set_xlabel('Year', fontsize=14)
        # set the y-axis label
        ax.set_ylabel('Scaling factor', fontsize=14)
        # Set the title of the plot
        ax.set_title(f'Scaling factor for {haz_type}', fontsize=16)
        # Set the legend of the plot
        ax.legend(title='Hazard set for given year', bbox_to_anchor=(1.05, 1), loc='upper left')
        # Show the plot
        plt.show()

    # Generate the hazard scaling dictionary
    haz_avail_dict = copy.deepcopy(haz_given_dict)

    return haz_avail_dict, haz_given_dict, haz_param_dict


#%% Generate the sample event IDs

def generate_sample_eventIDs(haz_given_dict, haz_param_dict, future_year, n_samples=100, sample_method='bayesian'):

    # Get smallest year
    start_year = min(haz_given_dict[list(haz_given_dict.keys())[0]].keys())


    # Store the distributions to use for sampling
    haz_Bayesian_select_dict = {}
    # Make a dictionary determing which hazard distribution to use for sampling
    if sample_method == 'bayesian':

        
        for haz_type in haz_param_dict.keys():
            # Create a data frame to store the distributions
            haz_Bayesian_select_dict[haz_type] = pd.DataFrame(index=range(n_samples), columns= range(start_year, future_year+1))
            # Decide on the distribution to use for sampling
            # Generate a zero to 1 vector with length of the number of samples    
            for year in range(start_year, future_year+1):
                # Generate a random vector
                rand_vec = np.random.rand(n_samples)
                # Get the years with values larger than 0
                temp_series = haz_param_dict[haz_type].loc[year]
                temp_series = temp_series[temp_series > 0]
                if len(temp_series) == 1:
                    haz_Bayesian_select_dict[haz_type].loc[:, year] = temp_series.index[0]
                else:
                # Loop over the number of samples and get th index of the value that is below the random number
                    for sample in range(n_samples):
                        if rand_vec[sample] <= temp_series.iloc[0]:
                            haz_Bayesian_select_dict[haz_type].loc[sample, year] = temp_series.index[0]
                        else:
                            haz_Bayesian_select_dict[haz_type].loc[sample, year] = temp_series.index[1]
                        

    #%% Create dummy impact objects for the sampled events

    haz_dummy_impact_dict = {}
    from climada.engine import Impact

    for haz_type, haz_dict in haz_given_dict.items():
        # Store a dictionary for each given hazard type
        haz_dummy_impact_dict[haz_type] = {}
        # Store a dummy impact object for each given year
        for given_year, haz in haz_dict.items():
            haz_dummy_impact_dict[haz_type][given_year] = Impact()
            haz_dummy_impact_dict[haz_type][given_year].at_event = haz.event_id
            haz_dummy_impact_dict[haz_type][given_year].frequency = haz.frequency
            

        

    #%% Make the sampling

    import climada.util.yearsets as yearsets

    sampled_eventIDs_dict = {}

    for haz_type in haz_param_dict.keys():
        # Store a data frame for each given year
        sampled_eventIDs_dict[haz_type] = {}
        # Get the years for which the hazard is given
        given_years = haz_given_dict[haz_type].keys()
        # Create a data frame to store the sampled events
        for given_year in given_years:
            sampled_eventIDs_dict[haz_type][given_year] = pd.DataFrame([[[] for _ in range(start_year, future_year+1)] for _ in range(n_samples)], 
                                                                index=range(n_samples), 
                                                                columns= range(start_year, future_year+1))
        # Check if bayesian sampling is used
        if sample_method == 'bayesian':
            # Loop over the number of samples
            for sample in range(n_samples):
                # Loop over the path years
                for path_year in range(start_year, future_year+1):
                    # Check which hazard distribution to use
                    haz_set_year = haz_Bayesian_select_dict[haz_type].loc[sample, path_year]
                    # Get the dummy impact object for the corresponding hazard set year
                    imp_dummy = haz_dummy_impact_dict[haz_type][haz_set_year]
                    # the number of years to sample impacts for (length(yimp.at_event) = sampled_years)
                    sampled_years = 1
                    # sample number of events per sampled year
                    lam = np.sum(imp_dummy.frequency)
                    events_per_year = yearsets.sample_from_poisson(sampled_years, lam)
                    # generate the sampling vector
                    sampling_vect = yearsets.sample_events(events_per_year, imp_dummy.frequency)
                    # Store the sampled event ids
                    sampled_eventIDs_dict[haz_type][haz_set_year].loc[sample, path_year] = sampling_vect
        # Check if frequency based sampling is used
        elif sample_method == 'frequency':
            # Loop over the number of samples
            for sample in range(n_samples):
                # Loop over the path years
                for path_year in range(start_year, future_year+1):
                    # For each given year
                    for given_year in given_years:
                        # Get the dummy impact object for the corresponding hazard set year
                        imp_dummy = haz_dummy_impact_dict[haz_type][given_year]
                        # the number of years to sample impacts for (length(yimp.at_event) = sampled_years)
                        sampled_years = 1
                        # Get the scale factor for the given year
                        scale_factor = haz_param_dict[haz_type].loc[path_year, given_year]
                        if scale_factor == 0:
                            continue
                        # sample number of events per sampled year
                        lam = np.sum(imp_dummy.frequency * scale_factor)
                        events_per_year = yearsets.sample_from_poisson(sampled_years, lam)
                        # generate the sampling vector
                        sampling_vect = yearsets.sample_events(events_per_year, imp_dummy.frequency*scale_factor)
                        # Store the sampled event ids
                        sampled_eventIDs_dict[haz_type][given_year].loc[sample, path_year] = sampling_vect

    return sampled_eventIDs_dict, haz_Bayesian_select_dict


#%% Generate the impact functions set mapping data frame per year

def generate_impfs_active_df(imp_fun_set_dict, future_year):
    # Get start year
    start_year = min(imp_fun_set_dict.keys())

    # Make the path years
    path_years = range(start_year, future_year+1)

    # Get the impact functions given years
    given_years_imp_fun = [year for year in imp_fun_set_dict.keys()]

    # Sorted given years
    sorted_given_years = sorted(given_years_imp_fun)

    # Make a data frame indicating which impact function set to use for each year
    impfs_active_df = pd.DataFrame(index=path_years, columns=['imp_fun_set'])

    # Fill the data frame
    current_imp_fun_set = sorted_given_years[0]
    for year in path_years:
        # Use the last impact function set if the year is after the last given year
        if year in sorted_given_years:
            current_imp_fun_set = year
        # Update the current impact function set
        impfs_active_df.loc[year, 'imp_fun_set'] = current_imp_fun_set

    # Create dictionary with the impact functions
    impfs_given_dict = copy.deepcopy(imp_fun_set_dict)
    impfs_avail_dict = copy.deepcopy(imp_fun_set_dict)


    return  impfs_avail_dict, impfs_given_dict, impfs_active_df


#%% Generate the adaptation measures set mapping data frame per year

def generate_meas_df(meas_dict, future_year, meas_inactive_years_dict=None):


    # Get start year
    start_year = min(meas_dict.keys())

    # Make the path years
    path_years = range(start_year, future_year+1)

    # Get the impact functions given years
    given_years = [year for year in meas_dict.keys()]

    # Sorted given years
    sorted_given_years = sorted(given_years)

    # Make a data frame indicating which impact function set to use for each year
    meas_active_df = pd.DataFrame(index=path_years, columns=['meas_idx_year'])

    # Fill the data frame
    current_measure_set = sorted_given_years[0]
    for year in path_years:
        # Use the last impact function set if the year is after the last given year
        if year in sorted_given_years:
            current_measure_set = year
        # Update the current impact function set
        meas_active_df.loc[year, 'meas_idx_year'] = current_measure_set


    #%% Add the columns for the adaptation measures

    # Get the unique measures and hazard types
    unique_measure_list = []
    haz_types_measure_list = []

    for given_year, measure_set in meas_dict.items():
        haz_types = list(measure_set.get_measure().keys())
        haz_types_measure_list += haz_types
        for haz_type in haz_types:
            unique_measure_list += list(measure_set.get_measure()[haz_type].keys())

    # Remove duplicates
    unique_measure_list = list(set(unique_measure_list))
    haz_types_measure_list = list(set(haz_types_measure_list))

    # Add the columns to the data frame
    for measure in unique_measure_list:
        meas_active_df[measure] = 1

    # Add the cells for the inactive measures to zero
    if meas_inactive_years_dict is not None:
        for measure, inactive_years in meas_inactive_years_dict.items():
            for year in inactive_years:
                meas_active_df.loc[year, measure] = 0

    # Crete teh measure active and aviable dict
    meas_avail_dict = copy.deepcopy(meas_dict)
    meas_given_dict = copy.deepcopy(meas_dict)

    return meas_avail_dict, meas_given_dict, meas_active_df


#%% Create the impact objects data frame mapping and unique impact objects data frame

def generate_imp_meas_df(exp_avail_dict, haz_avail_dict, impfs_active_df, exp_multipl_dict, haz_param_dict, meas_avail_dict, meas_active_df, incl_insurance = True):
    
    # Get the pathway years
    pathway_years = meas_active_df.index.get_level_values(0).unique().tolist()
    # Get the unique measures
    measure_names = meas_active_df.columns.get_level_values(0).unique().tolist()
    # remove 'measure_set' from the list
    measure_names.remove('meas_idx_year')
    # Add 'no measure' to the list and put first in the list
    measure_names.append('no measure')
    measure_names.sort(reverse=True)
    # Add 'insurance' to the list if it is included
    if incl_insurance:
        measure_names.append('insurance')
    # Get the exposure types
    exp_types = list(exp_avail_dict.keys())
    # Get the hazard types
    haz_types = list(haz_avail_dict.keys())
    
   # Define the columns of the unique impact objects data frame mappping
    columns = ['pathway_year',  'exp_type', 'exp_idx_year', 'exp_multiplier', 
            'haz_type', 'haz_idx_year', 'haz_multiplier', 
            'impfs_idx_year', 
            'meas_name', 'meas_is_active', 'meas_idx_year','meas_protects_haz_type', 
            'imp_obj_ID']

    # Create an empty data frame to store the unique impact objects
    imp_meas_map_year_df = pd.DataFrame(columns=columns)

    # Loop over the pathway years
    for path_year in pathway_years:

        # Get the impact function set index year
        impfs_idx_year = impfs_active_df.loc[path_year].values[0]

        ## Generate the impact objects data frame mapping
        # Loop over the exposure types
        for exp_type in exp_types:
                # Get the exposure index year
                exp_multi_df = exp_multipl_dict[exp_type]
                exp_multi_df = exp_multi_df.loc[path_year]
                exp_multi_df = exp_multi_df[exp_multi_df > 0]
                exp_idx_year = exp_multi_df.index[0]
                # Get the exposure multiplier
                exp_multiplier = exp_multi_df.values[0]

                # Loop over the hazard types
                for haz_type in haz_types:
                    # Get the hazard index years
                    haz_multi_df = haz_param_dict[haz_type]
                    haz_multi_df = haz_multi_df.loc[path_year]
                    haz_multi_df = haz_multi_df[haz_multi_df > 0]
                    haz_idx_years = haz_multi_df.index
                    # Get the hazard multipliers
                    haz_multipliers = haz_multi_df.values

                    # Loop over the different hazard index years
                    for haz_idx_year, haz_multiplier in zip(haz_idx_years, haz_multipliers):

                        # Create the core values dictionary to populate the data frame
                        core_values_dict = {'pathway_year': path_year, 'exp_type': exp_type, 'exp_idx_year': exp_idx_year, 'exp_multiplier': exp_multiplier, 
                            'haz_type': haz_type, 'haz_idx_year': haz_idx_year, 'haz_multiplier': haz_multiplier, 
                            'impfs_idx_year': impfs_idx_year, 
                            }

                        # Loop over the measures
                        for meas_name in measure_names:
                            # Get the measure is active
                            if meas_name == 'no measure' or meas_name == 'insurance':
                                meas_is_active = 1
                            else:
                                meas_is_active = meas_active_df.loc[path_year, meas_name]
                        
                            # Get the measure set idx year
                            meas_idx_year = meas_active_df.loc[path_year, 'meas_idx_year']

                            # Check if the measure protects the hazard type
                            if meas_name == 'no measure':
                                meas_protects_haz_type = 0
                            elif meas_name == 'insurance' or meas_name in meas_avail_dict[meas_idx_year].get_names()[haz_type]:
                                meas_protects_haz_type = 1
                            

                            # Create the unique impact object ID in the same order as the columns 
                            # Exclude for the pathway year, the multipliers, and the measure is active
                            imp_obj_ID = f'{exp_type}_{exp_idx_year}_{haz_type}_{haz_idx_year}_Impfs_{impfs_idx_year}_{meas_name}_{meas_idx_year}'

                            # If measure is inactive use the same impact object the as the no measure
                            if meas_name == 'no measure':
                                imp_obj_ID_no_meas = imp_obj_ID
                            elif meas_is_active == 0 or meas_protects_haz_type == 0:
                                imp_obj_ID = imp_obj_ID_no_meas

                            # Create a dictionary with the values and the columns as keys
                            meas_values_dict = {'meas_name': meas_name, 'meas_is_active': meas_is_active, 'meas_idx_year': meas_idx_year, 'meas_protects_haz_type': meas_protects_haz_type, 
                                        'imp_obj_ID': imp_obj_ID}
                            values_dict = {**core_values_dict, **meas_values_dict}

                            # Concatenate the values to the data frame
                            if imp_meas_map_year_df.empty: # If the data frame is empty create it
                                imp_meas_map_year_df = pd.DataFrame(values_dict, index=[0]) # Create the data frame
                            else:
                                imp_meas_map_year_df = pd.concat([imp_meas_map_year_df, pd.DataFrame(values_dict, index=[0])], ignore_index=True)

    ## Generate the unique impact objects data frame
    # Unique impact objects data frame
    imp_meas_unique_df = copy.deepcopy(imp_meas_map_year_df)
    # Drop pathway year, measure, the multipliers, and the measure is active
    imp_meas_unique_df = imp_meas_unique_df.drop(['pathway_year', 'meas_is_active', 'meas_protects_haz_type', 'exp_multiplier', 'haz_multiplier'], axis=1)
    # Drop duplicates
    imp_meas_unique_df = imp_meas_unique_df.drop_duplicates()


    return imp_meas_map_year_df, imp_meas_unique_df


#%% Utility function to generate the unique impact objects data frame

def create_filter_conditions(input_dict, param):
    filter_conditions = {}
    for key, value in input_dict.items():
        if not isinstance(value, list):
            value = [value]
        filter_conditions[key] = {param: value}
    return filter_conditions

def filter_dataframe(df, filter_conditions=None, derived_columns=None, base_cols=None):
    """
    This function filters a DataFrame based on provided conditions and calculates derived columns.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.
    - filter_conditions (dict): A dictionary specifying filtering conditions for columns.
    - derived_columns (dict): A dictionary specifying derived columns and their functions.

    Returns:
    - filtered_df (pandas.DataFrame): The filtered DataFrame based on conditions and derived columns.
    - boolean_df (pandas.DataFrame): A boolean DataFrame indicating whether values satisfy conditions.
    """

    # Create a copy of the input DataFrame
    filtered_df = df.copy()
    unfiltered_df = df.copy()
    
    # If conditions or derived columns are not provided, initialize them as empty dictionaries
    if filter_conditions is None:
        filter_conditions = {}
    
    if derived_columns is None:
        derived_columns = {}

    # Calculate and add derived columns to the filtered DataFrame
    if derived_columns:
        for new_col, function in derived_columns.items():
            filtered_df[new_col] = function(df)
            unfiltered_df[new_col] = function(df)

    # Create a boolean DataFrame to track conditions satisfaction
    if base_cols:
        boolean_df = df[base_cols].copy()
    else:
        boolean_df = df.copy()
    
    # Apply filtering conditions and update boolean DataFrame accordingly
    if filter_conditions:
        for col, cond in filter_conditions.items():
            
            if isinstance(cond, list):
                # Filter data based on whether column values are equal to the provided value
                filtered_df = filtered_df[filtered_df[col] == cond['equal']]
                boolean_df[col] = unfiltered_df[col] == cond['equal']
            elif 'equal' in cond:
                # Filter data based on whether column values are equal to the provided value
                filtered_df = filtered_df[filtered_df[col].isin(cond['equal'])]
                boolean_df[col] = unfiltered_df[col].isin(cond['equal'])
            elif 'in' in cond:
                # Filter data based on whether column values are in the provided list
                filtered_df = filtered_df[filtered_df[col].isin(cond['in'])]
                boolean_df[col] = unfiltered_df[col].isin(cond['in'])
            elif 'greater' in cond:
                # Filter data based on whether column values are greater than the provided value
                filtered_df = filtered_df[filtered_df[col] > cond['greater']]
                boolean_df[col] = unfiltered_df[col] > cond['greater']
            elif 'less' in cond:
                # Filter data based on whether column values are less than the provided value
                filtered_df = filtered_df[filtered_df[col] < cond['less']]
                boolean_df[col] = unfiltered_df[col] < cond['less']
            elif 'range' in cond:
                # Filter data based on whether column values are within the provided range
                lower, upper = cond['range']
                filtered_df = filtered_df[(filtered_df[col] >= lower) & (filtered_df[col] <= upper)]
                boolean_df[col] = (unfiltered_df[col] >= lower) & (unfiltered_df[col] <= upper)

    # Drop derived columns from the final filtered DataFrame
    if derived_columns:
        filtered_df = filtered_df.drop(derived_columns.keys(), axis=1)
     
    return filtered_df, boolean_df


#%% Generate the impact combo objects data frame

def generate_imp_combo_df(imp_meas_map_year_df, combo_dict = {}):
    '''
    Remove the no measure impact objects from the based on impact object IDs
    '''
            
    # If combo_dict is empty make a combination of all measures
    if not combo_dict:
        # Get all the unique measures
        meas_included = imp_meas_map_year_df['meas_name'].unique()
        # Create a dictionary with the combination of all measures
        combo_dict = {'All measures': meas_included}

    # Get the unique path years, exposure types, hazard types, and haz_idx_years 
    path_years = imp_meas_map_year_df['pathway_year'].unique()
    exp_types = imp_meas_map_year_df['exp_type'].unique()
    haz_types = imp_meas_map_year_df['haz_type'].unique()

    # Define the data frame to store the impact object mapping
    imp_combo_map_df = pd.DataFrame()

    # Create the impact object mapping for each combination of measures for each pathway year, exposure type, hazard type, and hazard index year

    # Loop over the pathway years
    for path_year in  path_years:
        # Loop over expsoire types
        for exp_type in exp_types:

            # Get the exposure index years
            exp_idx_year = imp_meas_map_year_df[(imp_meas_map_year_df['pathway_year'] == path_year) & (imp_meas_map_year_df['exp_type'] == exp_type)]['exp_idx_year'].unique()
            # Make as integer not array
            exp_idx_year = exp_idx_year[0]

            # Loop over hazard types
            for haz_type in haz_types:

                # Get the hazard index years
                haz_idx_years = imp_meas_map_year_df[(imp_meas_map_year_df['pathway_year'] == path_year) & (imp_meas_map_year_df['exp_type'] == exp_type) & (imp_meas_map_year_df['haz_type'] == haz_type)]['haz_idx_year'].unique()

                # Loop over hazard index years
                for haz_idx_year in haz_idx_years:

                    # Get the core values for the data frame
                    core_values_dict = {'pathway_year': path_year, 'exp_type': exp_type, 'exp_idx_year': exp_idx_year, 'haz_type': haz_type, 'haz_idx_year': haz_idx_year}

                    # Loop over the combinations of measures
                    for combo_name in combo_dict:
                        
                        # Get the measures in the combination
                        meas_in_combo = combo_dict[combo_name]

                        # Filter out the rows that satisfy the conditions in the data frame
                        filter_equal_conditions = create_filter_conditions(core_values_dict, 'equal')
                        filter_in_conditions = create_filter_conditions({'meas_name': meas_in_combo}, 'in')
                        filter_conditions = {**filter_equal_conditions, **filter_in_conditions}
                        sub_df = filter_dataframe(imp_meas_map_year_df, filter_conditions=filter_conditions)[0]

                        # Get list of the unique impact object IDs
                        based_on_Imp_obj_IDs = sub_df['imp_obj_ID'].unique()
                        if len(based_on_Imp_obj_IDs) == 0:
                            raise ValueError('No impact object found for the combination of measures')
                        # Exclude the no measure impact objects from the based on impact object IDs, if the no measure substring is in the impact object ID

                        # Split the list into two lists: one with 'no measure' and one without
                        no_measure_ids = [id for id in based_on_Imp_obj_IDs if 'no measure' in id]
                        other_ids = [id for id in based_on_Imp_obj_IDs if 'no measure' not in id]
                        # If there are IDs without 'no measure', return the unique ones
                        if other_ids:
                            other_ids = list(set(other_ids))
                            based_on_Imp_obj_IDs = other_ids
                        # If there are only 'no measure' IDs, return the unique ones and print a warning if there are duplicates
                        elif no_measure_ids:
                            unique_no_measure_ids = list(set(no_measure_ids))
                            if len(unique_no_measure_ids) > 1:
                                print("Warning: Duplicate 'no measure' IDs found")
                            based_on_Imp_obj_IDs = unique_no_measure_ids


                        # Create the unique impact object ID in the same order as the columns
                        # Exclude for the pathway year, the multipliers, and the measure is active
                        imp_obj_ID = f'{exp_type}_{haz_type}_{haz_idx_year}_{combo_name}'

                        # Create a dictionary with the values and the columns as keys
                        values_dict = {**core_values_dict, 'meas_name': combo_name, 'meas_included': [meas_in_combo], 'imp_obj_ID': imp_obj_ID, 'based_on_Imp_obj_IDs': [based_on_Imp_obj_IDs]}

                        # Concatenate the values to the data frame
                        if imp_combo_map_df.empty:
                            imp_combo_map_df = pd.DataFrame(values_dict, index=[0])
                        else:
                            imp_combo_map_df = pd.concat([imp_combo_map_df, pd.DataFrame(values_dict, index=[0])], ignore_index=True)


    # Get the unique impact object IDs
    # Generate the unique impact objects data frame
    # Only include the unique impact object IDs and the based on impact object IDs
    # Drop column 'pathway_year'
    imp_combo_unique_df = copy.deepcopy(imp_combo_map_df)
    # Drop column 'pathway_year' and 'meas_included'
    imp_combo_unique_df = imp_combo_unique_df.drop(columns=['pathway_year', 'meas_included'])
    # Convert lists in 'based_on_Imp_obj_IDs' to tuples
    imp_combo_unique_df['based_on_Imp_obj_IDs'] = imp_combo_unique_df['based_on_Imp_obj_IDs'].apply(tuple)
    # Drop duplicates
    imp_combo_unique_df = imp_combo_unique_df.drop_duplicates().reset_index(drop=True)
    # Convert tuples back to lists
    imp_combo_unique_df['based_on_Imp_obj_IDs'] = imp_combo_unique_df['based_on_Imp_obj_IDs'].apply(list) 

    return imp_combo_map_df, imp_combo_unique_df
#%% Generate the unique impact objects and store in a dictionary

def generate_imp_obj_dict(imp_meas_unique_df, exp_avail_dict, haz_avail_dict, impfs_avail_dict, meas_avail_dict, calc_rel_imp = True, imp_calc_params_kwargs= {'save_mat': True, 'assign_centroids': True, 'ignore_cover': False, 'ignore_deductible': False}):
    
    # Create a dictionary to store the unique impact objects
    imp_abs_dict = {}
    imp_rel_dict = {} # Store the relative impact objects (used when recovery rate  

    # Iterate over the unique impact objects rows in the data frame
    for row_df in imp_meas_unique_df.iterrows():
        # Get the exposure type
        exp_type = row_df[1]['exp_type']
        # Get the exposure index year
        exp_idx_year = row_df[1]['exp_idx_year']
        # Get the exposure object
        exp_obj = exp_avail_dict[exp_type][exp_idx_year]

        # Get the hazard type
        haz_type = row_df[1]['haz_type']
        # Get the hazard index year
        haz_idx_year = row_df[1]['haz_idx_year']
        # Get the hazard object
        haz_obj = haz_avail_dict[haz_type][haz_idx_year]

        # Get the impact function set index year
        impfs_idx_year = row_df[1]['impfs_idx_year']
        # Get the impact function set object
        impfs_obj = impfs_avail_dict[impfs_idx_year]

        # Get the measure
        meas_name = row_df[1]['meas_name']
        # Get the measure index year
        meas_idx_year = row_df[1]['meas_idx_year']
        # Get the measure object
        if meas_name == 'no measure':
            meas_obj = None
        elif meas_name == 'insurance':
            meas_obj = None
        else:
            meas_set = meas_avail_dict[meas_idx_year]
            meas_obj = meas_set.get_measure()[haz_type][meas_name]

        # Get the unique impact object ID
        imp_obj_ID = row_df[1]['imp_obj_ID']

        # Calculate the new exposure, impact function set, and hazard object given the measure
        if meas_obj is None:
            new_exp, new_impfs, new_haz = exp_obj, impfs_obj, haz_obj
        else:
            # Calculate the new exposure, impact function set, and hazard object given the measure
            new_exp, new_impfs, new_haz = meas_obj.apply(exp_obj, impfs_obj, haz_obj)
            # Check if the new exposure is the same as the old exposure
            if (new_exp.gdf != exp_obj.gdf).any(axis=1).any():
                raise ValueError('The new exposure should be the same as the old exposure. Otherwise, the realtiv impact matrix calculation will not work. Fix later update ')
        # Calculate the unique impact object
        imp_obj = ImpactCalc(new_exp, new_impfs, new_haz).impact(**imp_calc_params_kwargs)

        # Store the unique absolute impact object in the dictionary
        imp_abs_dict[imp_obj_ID] = imp_obj

        # Calculate the relative impact object (only works if the exposure is the same as the old exposure)
        if calc_rel_imp:
            # Create a dummy impact object
            exp_dummy = exp_obj.copy()
            exp_dummy.gdf.value = 1
            # Calculate the relative impact object
            imp_obj_rel = ImpactCalc(exp_dummy, new_impfs, new_haz).impact(save_mat= True, assign_centroids = True)
            # Store the unique relative impact object in the dictionary
            imp_rel_dict[imp_obj_ID] = imp_obj_rel

    return imp_abs_dict, imp_rel_dict
