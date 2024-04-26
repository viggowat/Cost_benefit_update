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

exp_avail_dict, exp_given_dict, exp_multipl_dict, exp_intrpl_dict = sfc.generate_exp_sets(exp_dict, intr_param,  future_year=year_future, growth_rate=0.02)

# %% [markdown]
# ## Step 2 - Create the hazard container

# %%
# Parameters
# Input to function
year_0 = START_YEAR
year_1 = year_0+15
year_2 = year_1+5
year_3 = year_2 + 5

# Hazard set TC
haz_dict = {}
haz_dict[year_0] = Hazard.from_hdf5(HAZ_DEMO_H5)
haz_dict[year_1] = copy.deepcopy(haz_dict[year_0])
haz_dict[year_1].intensity *= 1.5
haz_dict[year_2] = copy.deepcopy(haz_dict[year_0])
haz_dict[year_2].intensity *= 2
haz_dict[year_3] = copy.deepcopy(haz_dict[year_0])
haz_dict[year_3].intensity *= 3

#
# %% [markdown]
# ### Generate hazard attributes

# %%
# Parameters
year_future = FUTURE_YEAR
intr_param = 0.4

haz_avail_dict, haz_given_dict, haz_param_dict = sfc.generate_haz_sets(haz_dict, intr_param, future_year=FUTURE_YEAR)

# %% [markdown]
# ## Step 3 - Sample the event IDs

# %%
# Decide on sampling method
# Bayesian, or frequency based
sample_method = 'frequency' # 'bayesian' or 'frequency'

# Number of samples
n_samples = 100

sampled_eventIDs_dict, haz_Bayesian_select_dict = sfc.generate_sample_eventIDs(haz_avail_dict, haz_param_dict, future_year=FUTURE_YEAR, n_samples=n_samples, sample_method=sample_method)

# %% [markdown]
# ## Step 4 - Create the impact fun sets active df

# %%
# Set the impact functions per year
imp_fun_set_dict = {} 
for year in [year_0, year_0+4, year_0+10]:
    imp_fun_set_dict[year] = Entity.from_excel(ENT_DEMO_TODAY).impact_funcs

impfs_avail_dict, impfs_given_dict, impfs_active_df = sfc.generate_impfs_active_df(imp_fun_set_dict, future_year=FUTURE_YEAR)

# %% [markdown]
# ## Step 5 - Create the adapdation measures active df

# %%
# Define parameter adaptation measure dictionary
meas_dict = {}
for year in [year_0, year_0+3, year_0+12]:
    measure_set = Entity.from_excel(ENT_DEMO_TODAY).measures
    meas_dict[year] = copy.deepcopy(measure_set)
# Define parameter adaptation measure off
meas_inactive_years_dict = {'Seawall': [year_0+5, year_0+10], 'Building code': [year_0+1, year_0+12]}

# Generate the measure active dataframe
meas_avail_dict, meas_given_dict, meas_active_df = sfc.generate_meas_df(meas_dict, future_year= FUTURE_YEAR, meas_inactive_years_dict=meas_inactive_years_dict)

# %% [markdown]
# ## Step 6 - Create the impact objects
# 
# - Remember, for every given of an exposure, adaptation measure, impact_function_set and TC there is a unique impact object 
# - Remember that if a measure is inactive it is the same as having no measure to sample from. 
#     - For combinations of measure you then need to exclude sampling from that object.



# %% Create a function to generate a data frame of the unique impact objects
# Make a function that gets the unique impact objects for a given year, hazard, exposure, measure and impact function set, etc ... 

# Get all the necessary data frames

# Rmeber to add insurance as measuer
incl_insurance = True

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



#%% Create the unique impact objects data frame mapping 
# Make as a function 

# Define the columns of the unique impact objects data frame mappping
columns = ['pathway_year',  'exp_type', 'exp_idx_year', 'exp_multiplier', 'haz_type', 'haz_idx_year', 'haz_multiplier', 'impfs_idx_year', 'meas_name', 'meas_is_active', 'meas_idx_year','meas_protects_haz_type','imp_obj_ID']

# Create an empty data frame to store the unique impact objects
imp_obj_map_df = pd.DataFrame(columns=columns)

# Loop over the pathway years
for path_year in pathway_years:

    # Get the impact function set index year
    impfs_idx_year = impfs_active_df.loc[path_year].values[0]


    for exp_type in exp_types:
            # Get the exposure index year
            exp_multi_df = exp_multipl_dict[exp_type]
            exp_multi_df = exp_multi_df.loc[path_year]
            exp_multi_df = exp_multi_df[exp_multi_df > 0]
            exp_idx_year = exp_multi_df.index[0]
            # Get the exposure multiplier
            exp_multiplier = exp_multi_df.values[0]

            for haz_type in haz_types:
                # Get the hazard index years
                haz_multi_df = haz_param_dict[haz_type]
                haz_multi_df = haz_multi_df.loc[path_year]
                haz_multi_df = haz_multi_df[haz_multi_df > 0]
                haz_idx_years = haz_multi_df.index
                # Get the hazard multipliers
                haz_multipliers = haz_multi_df.values


                for meas_name in measure_names:
                    # Get the measure is active
                    if meas_name == 'no measure' or meas_name == 'insurance':
                        meas_is_active = 1
                    else:
                        meas_is_active = meas_active_df.loc[year, meas_name]
                
                    # Get the measure set idx year
                    meas_idx_year = meas_active_df.loc[path_year, 'meas_idx_year']

                    # Check if the measure protects the hazard type
                    if meas_name == 'no measure':
                        meas_protects_haz_type = 0
                    elif meas_name == 'insurance' or meas_name in meas_avail_dict[meas_idx_year].get_names()[haz_type]:
                        meas_protects_haz_type = 1
                    

                    # Create the unique impact object ID in the same order as the columns 
                    # Exclude for the pathway year, the multipliers, and the measure is active
                    imp_obj_ID = f'{exp_type}_{exp_idx_year}_{haz_type}_{haz_idx_years[0]}_Impfs_{impfs_idx_year}_{meas_name}_{meas_idx_year}'

                    # If measure is inactive use the same impact object the as the no measure
                    if meas_name == 'no measure':
                        imp_obj_ID_no_meas = imp_obj_ID
                    elif meas_is_active == 0 or meas_protects_haz_type == 0:
                        imp_obj_ID = imp_obj_ID_no_meas

                    # Create a dictionary with the values and the columns as keys
                    values = [path_year, exp_type, exp_idx_year, exp_multiplier, haz_type, haz_idx_years[0], haz_multipliers[0], impfs_idx_year, meas_name, meas_is_active, meas_idx_year, meas_protects_haz_type, imp_obj_ID]
                    values_dict = dict(zip(columns, values))

                    # Concatenate the values to the data frame
                    if imp_obj_map_df.empty:
                        imp_obj_map_df = pd.DataFrame(values_dict, index=[0])
                    else:
                        imp_obj_map_df = pd.concat([imp_obj_map_df, pd.DataFrame(values_dict, index=[0])], ignore_index=True)

# Unique impact objects data frame
imp_obj_unique_df = copy.deepcopy(imp_obj_map_df)
# Drop pathway year, measure, the multipliers, and the measure is active
imp_obj_unique_df = imp_obj_unique_df.drop(['pathway_year', 'meas_is_active', 'meas_protects_haz_type', 'exp_multiplier', 'haz_multiplier'], axis=1)
# Drop duplicates
imp_obj_unique_df = imp_obj_unique_df.drop_duplicates()


#%% Calculate the unique impact objects and store in a dictionary
# Make as a function

from climada.engine import ImpactCalc

# Create a dictionary to store the unique impact objects
imp_obj_dict = {}

# Iterate over the unique impact objects rows in the data frame
for row_df in imp_obj_unique_df.iterrows():
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

    # Calculate the unique impact object
    imp_obj = ImpactCalc(new_exp, new_impfs, new_haz).impact()



    

# %%
