

#%% Import the necessary libraries
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

import submodules.utils as utls


#%% Default functions

def EXP_EXPNL_GRWTH(exp_dict, year, growth_rate=0.04):
    year_start = list(exp_dict.keys())[0]
    exp_temp = copy.deepcopy(exp_dict[year_start])
    # Apply the exponential growth rate to the exposure value at each exposure point
    exp_temp.gdf.value = exp_temp.gdf.value * (1+growth_rate)**(year-year_start)
    return exp_temp

GROWTH_RATE = 0.02



#%% Generate the interpolated/extrapolated exposure sets

def generate_exp_sets(exp_dict, intr_param=1,  future_year=None, unique_identifiers_list = [], growth_rate=GROWTH_RATE):
    """
    Generate interpolated and extrapolated exposure sets per year
    """

    # Start year
    start_year = min(exp_dict.keys())

    # Check if the future_year is None
    if future_year is None:
        future_year = max(exp_dict.keys())

    # Make all the items in the dictionary as list
    for key, itm in exp_dict.items():
        if not isinstance(itm, list):
            exp_dict[key] = [itm]

    # Get the unique exposure value_unit
    exp_given_dict = _get_unique_exposures_per_value_unit(exp_dict, start_year)

    # Get the unique exposure identifiers
    exp_unique_ids_dict = _get_unique_exposure_identifiers(exp_given_dict, unique_identifiers_list)

    # Get the intersection of the unique exposure identifiers
    exp_given_mod_dict = _add_non_existing_unique_exposure_ids(exp_given_dict, exp_unique_ids_dict)

    # Get the interpolate the exposure per year
    exp_inter_pol_years = _generate_inter_pol_years(exp_given_mod_dict)

    # Generate the interpolated and extrapolated exposure sets
    exp_avail_dict = _generate_exp_sets(exp_given_mod_dict, exp_inter_pol_years, intr_param,  future_year=future_year, growth_rate=growth_rate)

    # Calculate the scaling factor for each year
    exp_multipl_dict = _generate_scale_exp(exp_avail_dict, future_year, growth_rate)

    return exp_avail_dict, exp_given_mod_dict, exp_multipl_dict, exp_inter_pol_years, exp_unique_ids_dict



#%%% Plot the interpolated/extrapolated exposure value at a given or random exposure point

def plot_exp_value(exp_avail_dict, exp_given_mod_dict, exp_multipl_dict, future_year, exp_point=None):

    # Plot the interpolated/extrapolated exposure value at each exposure point      
    for value_unit, exp_dict in exp_avail_dict.items():
        # Plot the interpolated/extrapolated exposure value at each exposure point
        # Plot random exposure point for each year
        # Get a random exposure point from first year
        first_year = list(exp_dict.keys())[0]
        # Get a random exposure point
        if exp_point is None:
            exp_point = np.random.randint(0, len(exp_dict[first_year].gdf))
        else:
            exp_point = exp_point
            if exp_point >= len(exp_dict[first_year].gdf):
                raise ValueError('The exposure point is not in the exposure set')
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
        given_years = list(exp_given_mod_dict[value_unit].keys())
        given_values = [exp_given_mod_dict[value_unit][year].gdf.value[exp_point] for year in given_years]
        plt.scatter(given_years, given_values, color='blue', label='Given exposure value')
        plt.xlabel('Year')
        plt.ylabel('Exposure value')
        plt.legend()
        plt.title(f' For value_unit {value_unit} - Value at random exposure point (idx= {exp_point}) for each year') 
        plt.show()


#%% Get the unique exposure value_unit

def _get_unique_exposures_per_value_unit(exp_dict, start_year):

    # Store all the available exposure sets in a dictionary
    exp_given_dict = {} 

    # Check all the possible exposure value_unit and store them in a list
    exp_value_units = []

    # Check all the possible exposure value_unit, referring to the exposure types
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
        
    return exp_given_dict
            

#%% Get the unique exposure identifiers

def _get_unique_exposure_identifiers(exp_given_dict, unique_identifiers_list =[]):

    # Add  ['latitude', 'longitude'] identifiers to the list if not already in the list
    if 'latitude' not in unique_identifiers_list:
        unique_identifiers_list.append('latitude')
    if 'longitude' not in unique_identifiers_list:
        unique_identifiers_list.append('longitude')

    # Make a base exp unique identifier data frame
    base_exp_unique_ids_df = pd.DataFrame(columns=unique_identifiers_list)

    # Get the unique exposure value_unit
    exp_value_units = list(exp_given_dict.keys())

    # Make a dictionary where you store the unique exposure identifiers for each value_unit
    exp_unique_ids_dict = {value_unit: base_exp_unique_ids_df.copy() for value_unit in exp_value_units}

    # For each value_unit get the unique exposure IDs
    for value_unit, exp_dict in exp_given_dict.items():

        # Get the unique exposure IDs
        for year, exp in exp_dict.items():
            # Get the gdf of the exposure
            gdf = exp.gdf

            # Get the unique exposure IDs
            cols_in_df = [col for col in  gdf.columns if col in unique_identifiers_list]
            unique_ids = gdf[cols_in_df].drop_duplicates()
            # Get the columns that are not in the data frame
            cols_not_in_df = [col for col in unique_identifiers_list if col not in cols_in_df]
            # Set the columns that are not in the data frame to None
            unique_ids[cols_not_in_df] = None

            # Add to the data frame
            if exp_unique_ids_dict[value_unit].empty:
                exp_unique_ids_dict[value_unit] = unique_ids

            else:
                exp_unique_ids_dict[value_unit] = pd.concat([exp_unique_ids_dict[value_unit], unique_ids], ignore_index=True)

        # Drop duplicates
        exp_unique_ids_dict[value_unit] = exp_unique_ids_dict[value_unit].drop_duplicates()

        # Reset index
        exp_unique_ids_dict[value_unit].reset_index(drop=True, inplace=True)

        # For columns with all None values drop the columns
        exp_unique_ids_dict[value_unit] = exp_unique_ids_dict[value_unit].dropna(axis=1, how='all')

    return exp_unique_ids_dict



#%% Get the intersection of the unique exposure identifiers

def _add_non_existing_unique_exposure_ids(exp_given_dict, exp_unique_ids_dict):

    # Make a deep copy of the exp_given_dict
    exp_given_mod_dict = copy.deepcopy(exp_given_dict)

    # For each value_unit get the unique exposure geo locations
    for value_unit, exp_dict in exp_given_mod_dict.items():

        # Get the unique exposure IDs data frame
        unique_ids_df = exp_unique_ids_dict[value_unit]
        cols_in_df = list(unique_ids_df.columns)

        for year, exp in exp_dict.items():
            # Get the gdf of the exposure
            gdf = exp.gdf

            # See if any of the unique exposure IDs are not in the exposure set
            merged_df = pd.merge(unique_ids_df, gdf, on=cols_in_df, how='outer', indicator=True)

            # Get the unique exposure IDs that are not in the exposure set
            missing_ids = merged_df[merged_df['_merge'] == 'left_only'][cols_in_df]

            if not missing_ids.empty:
                print(f'The missing unique exposure IDs in the year {year} for exposure type {value_unit} are:')
                print(missing_ids)
                # Add the missing unique exposure IDs to the exposure set and set the value to 0
                for idx, row in missing_ids.iterrows():
                    # Get the first row of the exposure set
                    first_row = gdf.iloc[0]
                    # Create a new row with the missing unique exposure IDs
                    new_row = first_row.copy()
                    new_row[cols_in_df] = row
                    new_row['value'] = 0
                    # Add the dedcutible and coverage if they exist
                    if 'deductible' in gdf.columns:
                        new_row['deductible'] = 0
                    if 'cover' in gdf.columns:
                        new_row['cover'] = 0
                    # Add the new row to the exposure set without using append
                    gdf.loc[len(gdf)] = new_row

            # Reorder the rows of the gdf to have the same rows and index as in unique_ids_df without adding new columns from unique_ids_df
            gdf = pd.merge(unique_ids_df, gdf, on=cols_in_df, how='left')

            # Check so that the number of rows in the exposure set is the same as the number of unique exposure IDs
            if len(gdf) != len(unique_ids_df):
                raise ValueError(f'The number of rows in the exposure set is not the same as the number of unique exposure IDs in the year {year}')
            
            # Update the exposure set
            exp.gdf = gdf

    return exp_given_mod_dict




def _generate_exp_per_year(exp_dict, intr_param, year_future, exp_expl_fnc=EXP_EXPNL_GRWTH):
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
                exp_temp.gdf.value[idx] = utls.interpolate_value(year, [year_start, year_future], [exp_start_val, exp_future_val], intr_param)
            # Add the exposure to the dictionary
            exp_per_year_dict[year] = exp_temp
    # If only one exposure year, extrapolate the exposure value at each exposure point
    else:
        # Extrapolate the exposure value at each exposure point
        for year in range(year_start,year_future +1):
            # Add the exposure to the dictionary
            exp_per_year_dict[year] = exp_expl_fnc(exp_dict, year)

    return exp_per_year_dict

#%%% Generate the interppolated Exposure dictionary

def _generate_inter_pol_years(exp_given_mod_dict):

    # Get the year range for each value_unit
    exp_inter_pol_years = {}
    # Count the number of exposure objects for each value_unit
    for value_unit, exp_dict in exp_given_mod_dict.items():
        exp_inter_pol_years[value_unit] = None
        print(f'The number of exposure objects for {value_unit} is {len(exp_dict)}')
        # Get the year range for each value_unit
        exp_inter_pol_years[value_unit] = [min(exp_dict.keys()), max(exp_dict.keys())]

    return exp_inter_pol_years

#%% Generate the interpolated and extrapolated exposure sets

def _generate_exp_sets(exp_given_mod_dict, exp_inter_pol_years, intr_param,  future_year=None, growth_rate=0.02):

    # Store all the interpolated and extrapolated exposure sets
    exp_avail_dict = {} # Store all the interpolated and extrapolated exposure sets 

    # Get the unique exposure value_unit
    exp_value_units = list(exp_given_mod_dict.keys())

    # Interpolate the exposure sets
    for value_unit in exp_value_units:
        exp_avail_dict[value_unit] = {}
        # Check if the value_unit has more than one exposure set
        if not exp_inter_pol_years[value_unit]:
            exp_avail_dict[value_unit] = exp_given_mod_dict[value_unit]
        else:
            exp_dict = exp_given_mod_dict[value_unit]
            # Interpolate
            for year_start, year_end in zip(list(exp_dict.keys())[:-1], list(exp_dict.keys())[1:]):
                #print(f'The pair of years is {year_start} and {year_end}')
                # Make a subset of the exposure dictionary
                exp_dict_subset = {}
                exp_dict_subset[year_start] = exp_dict[year_start]
                exp_dict_subset[year_end] = exp_dict[year_end]
                # Interpolation parameter
                exp_temp_dict = _generate_exp_per_year(exp_dict_subset, intr_param, future_year)
                # Add the interpolated exposure to the dictionary
                exp_avail_dict[value_unit].update(exp_temp_dict)

    # Check so that for the given years in exp_given_mod_dict the gdf is the same as in exp_avail_dict
    for value_unit, exp_dict in exp_given_mod_dict.items():
        for year in exp_dict.keys():
            # get exposure from the given and available exposure sets
            gdf_given = exp_dict[year].gdf
            gdf_avail = exp_avail_dict[value_unit][year].gdf

            if not gdf_given.equals(gdf_avail):
                # Print the difference
                print(f'The difference in the gdf for {value_unit} in year {year} is:')
                print(gdf_given[~gdf_given.isin(gdf_avail)].dropna())
                print(gdf_avail[~gdf_avail.isin(gdf_given)].dropna())
                #raise ValueError('The gdf is not the same in the given and available exposure sets')

    return exp_avail_dict

#%%% Generate the scale Exposure dataframe dictionary

def _generate_scale_exp(exp_avail_dict, future_year, growth_rate):

    # Get the start year
    start_year = min(exp_avail_dict[list(exp_avail_dict.keys())[0]].keys())

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

    return exp_multipl_dict

