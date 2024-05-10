# %%
#%% Import packages

import matplotlib.pyplot as plt
import copy
import pandas as pd
import numpy as np

from climada.entity import Entity
from climada.util.constants import ENT_DEMO_TODAY, ENT_DEMO_FUTURE, HAZ_DEMO_H5
from climada.hazard import Hazard

import steps_functions as sfc

# %% [markdown]
# ## Code structure

# %% [markdown]
# Based on unique CLIMADA objects (exposure, measure, … ) at
# - Given years: defined objects by the user at given years
# - Available years: derived unique objects
# - Pathway years: the year we estimate the risk metric

# %% [markdown]
# ### Inital parameters

# %%
START_YEAR = 2020
FUTURE_YEAR = 2070


# %% [markdown]
# ## Step 1 - Organize the Exposure objects

# %% [markdown]
# ### Create a exposure dictionary

# %%
# Parameters
# Input to function
year_0 = START_YEAR
year_1 = year_0+5
year_2 = year_1+5
year_3 = year_2 + 25
year_4 = year_3 + 5



ent_USD = Entity.from_excel(ENT_DEMO_TODAY).exposures
ent_USD.gdf['value_unit'] = 'USD'
ent_PEOPLE = Entity.from_excel(ENT_DEMO_TODAY).exposures
ent_PEOPLE.gdf['value_unit'] = 'PEOPLE'

people_growth_rate = 0.01
usd_growth_rate = 0.02

# Build the exposure dictionary
exp_dict = {}
exp_dict[year_0] = [ent_USD, ent_PEOPLE]
# Give value_unit to the exposure 'USD'
exp_dict[year_1] = [copy.deepcopy(ent_USD), copy.deepcopy(ent_PEOPLE)]
exp_dict[year_1][0]
exp_dict[year_1][0].gdf.value *= (1+people_growth_rate)**(year_1-year_0)
exp_dict[year_1][1].gdf.value *= (1+usd_growth_rate)**(year_1-year_0)
# Give value_unit to the exposure 'USD'
exp_dict[year_2] = [copy.deepcopy(ent_USD), copy.deepcopy(ent_PEOPLE)]
exp_dict[year_2][0].gdf.value *= (1+people_growth_rate)**(year_2-year_0)
exp_dict[year_2][1].gdf.value *= (1+usd_growth_rate)**(year_2-year_0)
# Give value_unit to the exposure 'USD'
exp_dict[year_3] = [copy.deepcopy(ent_USD)]
exp_dict[year_3][0].gdf.value *= (1+usd_growth_rate)**(year_3-year_0)
# Give value_unit to the exposure 'USD'
exp_dict[year_4] = [copy.deepcopy(ent_USD)]
exp_dict[year_4][0].gdf.value *= (1+usd_growth_rate)**(year_4-year_0)
# Add a new exposure point with slightly different location
exp_dict[year_3][0].gdf.iloc[-1, exp_dict[year_3][0].gdf.columns.get_loc('longitude')] = exp_dict[year_3][0].gdf.iloc[-1, exp_dict[year_3][0].gdf.columns.get_loc('longitude')] + 1.0


# Remove second item of the list in year_3
#exp_dict[year_0] = [exp_dict[year_0][0]]

# %% [markdown]
# ### Generate expsoure function

# %%

# Parameters
growth_rate = 0.08
year_future = FUTURE_YEAR
intr_param = 1


# exp_avail_dict, exp_given_dict, exp_multipl_dict, exp_intrpl_dict = sfc.generate_exp_sets(exp_dict, intr_param,  future_year=year_future, growth_rate=0.02)

#%%

# Make he parameters
future_year=year_future
exp_dict = exp_dict
intr_param = intr_param
growth_rate = growth_rate


#%% Functions related to the exposure objects

# Define the function generate_exp_per_year that generates interpolated and extrapolated exposure sets per year
#def generate_exp_sets(exp_dict, intr_param,  future_year=None, growth_rate=0.02):
"""
Generate interpolated and extrapolated exposure sets per year
"""

start_year = min(exp_dict.keys())

#%%% Generate the Exposure given dictionary and update so all the exposure sets have the same geo locations

#%% Initial checks

# Check if the future_year is None
if future_year is None:
    future_year = max(exp_dict.keys())


# Make all the items in the dictionary as list
for key, itm in exp_dict.items():
    if not isinstance(itm, list):
        exp_dict[key] = [itm]




#%% Get the unique exposures per value_unit

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
        
    return exp_value_units, exp_given_dict

exp_value_units, exp_given_dict = _get_unique_exposures_per_value_unit(exp_dict, start_year)


# # Check all the possible exposure value_unit, refreing to the exposure types
# for year, exp_list in exp_dict.items():
#     # Check if the column value_unit exists
#     for exp in exp_list:
#         # If the column value_unit does not exist, raise an error
#         if 'value_unit' not in exp.gdf.columns:
#             raise ValueError('The column value_unit should exist in the exposure dataframe.')
#         # If the column value_unit exists, ...
#         else:
#             value_unit = exp.gdf.value_unit.unique()
#             # Check if the value_unit column has only one unique value
#             if len(value_unit) > 1:
#                 raise ValueError('The value_unit column should have only one unique value')
#             # Check if the value_unit is not already in the list
#             elif value_unit[0] not in exp_value_units:
#                 # Store the value_unit in the list
#                 exp_value_units.append(value_unit[0])
#                 # Create a new key in the exp_given_dict dictionary
#                 exp_given_dict[value_unit[0]] = {}
#             # Store the exposure in the dictionary but check if the year is already in the dictionary
#             if year in exp_given_dict[value_unit[0]]:
#                 raise ValueError('The year already exists in the exposure set')
#             # Store the exposure in the dictionary
#             exp_given_dict[value_unit[0]][year] = exp

# # Check that for the first year the number of exposure objects is the same
# for value_unit, exp_dict in exp_given_dict.items():
#     # Check if the number of exposure objects is the same
#     if not start_year in exp_dict.keys():
#         raise ValueError('Both exposure objects should exist for the first year')



            

#%% Get the unique exposure identifiers

def _get_unique_exposure_identifiers(exp_dict, unique_identifiers_list = ['latitude', 'longitude']):

    # Make a base exp unique identifier data frame
    base_exp_unique_ids_df = pd.DataFrame(columns=unique_identifiers_list)

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

exp_unique_ids_dict = _get_unique_exposure_identifiers(exp_dict, unique_identifiers_list = ['latitude', 'longitude'])

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


exp_given_mod_dict = _add_non_existing_unique_exposure_ids(exp_given_dict, exp_unique_ids_dict)


#%% Default functions

def exp_expnl_growth(exp_dict, year, growth_rate=0.04):
    year_start = list(exp_dict.keys())[0]
    exp_temp = copy.deepcopy(exp_dict[year_start])
    # Apply the exponential growth rate to the exposure value at each exposure point
    exp_temp.gdf.value = exp_temp.gdf.value * (1+growth_rate)**(year-year_start)
    return exp_temp

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

def _generate_exp_per_year(exp_dict, intr_param, year_future, exp_expl_fnc=exp_expnl_growth):
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

    return exp_per_year_dict

#%%% Generate the interppolated Exposure dictionary
            
# Get the year range for each value_unit
exp_inter_pol_years = {}
# Count the number of exposure objects for each value_unit
for value_unit, exp_dict in exp_given_mod_dict.items():
    exp_inter_pol_years[value_unit] = None
    print(f'The number of exposure objects for {value_unit} is {len(exp_dict)}')
    # Get the year range for each value_unit
    exp_inter_pol_years[value_unit] = [min(exp_dict.keys()), max(exp_dict.keys())]

#%% Generate the interpolated and extrapolated exposure sets

def _generate_exp_sets(exp_given_mod_dict, exp_inter_pol_years, intr_param,  future_year=None, growth_rate=0.02):

    # Store all the interpolated and extrapolated exposure sets
    exp_avail_dict = {} # Store all the interpolated and extrapolated exposure sets 

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

    return exp_avail_dict

exp_avail_dict = _generate_exp_sets(exp_given_mod_dict, exp_inter_pol_years, intr_param,  future_year=year_future, growth_rate=growth_rate)

#%%% Generate the scale Exposure dataframe dictionary

def _generate_scale_exp(exp_avail_dict, future_year, growth_rate):
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

# Calculate the scaling factor for each year
exp_multipl_dict = _generate_scale_exp(exp_avail_dict, future_year, growth_rate)

# # Make a dictionary where you store the scaleing factor for each year in a data frame
# exp_multipl_dict = {}
# # Calculate the scaling factor for each year
# for value_unit, exp_dict in exp_avail_dict.items():
#     # Make a diagonal data frame with zeros and ones in the diagonal
#     exp_multipl_dict[value_unit] = pd.DataFrame(index=range(start_year, future_year+1), columns=exp_dict.keys(), dtype=float)
#     for year_row in exp_multipl_dict[value_unit].index:
#         for year_col in exp_multipl_dict[value_unit].columns:
#             if year_row == year_col:
#                 exp_multipl_dict[value_unit].loc[year_row, year_col] = 1.0
#             else:
#                 exp_multipl_dict[value_unit].loc[year_row, year_col] = 0.0

#     # Get the max column year
#     max_col_year = max(exp_dict.keys())
#     # If the max column year is not the last year, extrapolate the values from one and add them to the last year
#     if max_col_year != future_year:
#         # Extrapolate the exposure value at each exposure point
#         for year_row in range(max_col_year,future_year+1):
#             # Add the exposure to the dictionary
#             exp_multipl_dict[value_unit].loc[year_row, max_col_year] = (1 + growth_rate)**(year_row - max_col_year)

#%%% Plot the interpolated/extrapolated exposure value at a given or random exposure point


def _plot_exp_value(exp_avail_dict, exp_given_mod_dict, exp_multipl_dict, future_year, exp_point=None):

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

# return exp_avail_dict, exp_given_dict, exp_multipl_dict, exp_inter_pol_years