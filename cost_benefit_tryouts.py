#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:33:55 2024

@author: vwattin
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd

from climada.entity.measures import Measure, MeasureSet
from climada.entity import DiscRates
from climada.entity import ImpactFuncSet, ImpfTropCyclone
from climada.entity import Entity


from climada.engine import ImpactCalc
from climada.engine import CostBenefit
from climada.engine.cost_benefit import risk_aai_agg

from climada.util.api_client import Client


#%% Initialize 

#%%% Download hazard

client = Client()
future_year = 2080
haz_present = client.get_hazard('tropical_cyclone', 
                                properties={'country_name': 'Haiti', 
                                            'climate_scenario': 'historical',
                                            'nb_synth_tracks':'10'})
haz_future = client.get_hazard('tropical_cyclone', 
                                properties={'country_name': 'Haiti', 
                                            'climate_scenario': 'rcp60',
                                            'ref_year': str(future_year),
                                            'nb_synth_tracks':'10'})

#%%% Define impact function

impf_tc = ImpfTropCyclone.from_emanuel_usa()

# add the impact function to an Impact function set
impf_set = ImpactFuncSet([impf_tc])
impf_set.check()

#%%% Download LitPop economic exposure data

exp_present = client.get_litpop(country='Haiti')

exp_future = copy.deepcopy(exp_present)
exp_future.ref_year = future_year
n_years = exp_future.ref_year - exp_present.ref_year + 1
growth_rate = 1.02
growth = growth_rate ** n_years
exp_future.gdf['value'] = exp_future.gdf['value'] * growth


# This is more out of politeness, since if there's only one impact function
# and one `impf_` column, CLIMADA can figure it out
exp_present.gdf.rename(columns={"impf_": "impf_TC"}, inplace=True)
exp_present.gdf['impf_TC'] = 1
exp_future.gdf.rename(columns={"impf_": "impf_TC"}, inplace=True)
exp_future.gdf['impf_TC'] = 1

#%%% Define adaptation measures

meas_1 = Measure(
    haz_type='TC',
    name='Measure A',
    color_rgb=np.array([0.8, 0.1, 0.1]),
    cost=5000000000,
    hazard_inten_imp=(1, -5),     # Decrease wind speeds by 5 m/s
    risk_transf_cover=0,
)

meas_2 = Measure(
    haz_type='TC',
    name='Measure B',
    color_rgb=np.array([0.1, 0.1, 0.8]),
    cost=220000000,
    paa_impact=(1, -0.10),   # 10% fewer assets affected
)

# gather all measures
meas_set = MeasureSet(measure_list=[meas_1, meas_2])
meas_set.check()


#%%% Define discount rates


year_range = np.arange(exp_present.ref_year, exp_future.ref_year + 1)
annual_discount_zero = np.zeros(n_years)
annual_discount_stern = np.ones(n_years) * 0.014

discount_zero = DiscRates(year_range, annual_discount_zero)
discount_stern = DiscRates(year_range, annual_discount_stern)


#%%% Create the entities

entity_present_disc = Entity(exposures=exp_present, disc_rates=discount_stern,
                             impact_func_set=impf_set, measure_set=meas_set)
entity_future_disc = Entity(exposures=exp_future, disc_rates=discount_stern,
                            impact_func_set=impf_set, measure_set=meas_set)

#%%% Calc cost_benefit

costben_disc = CostBenefit()
costben_disc.calc(haz_present, entity_present_disc, haz_future=haz_future, ent_future=entity_future_disc,
                  future_year=future_year, risk_func=risk_aai_agg, imp_time_depen=1, save_imp=True)
print(costben_disc.imp_meas_future['no measure']['impact'].imp_mat.shape)


ax = costben_disc.plot_cost_benefit()


#%%% Combined measure
combined_costben_disc = costben_disc.combine_measures(['Measure A', 'Measure B'],
                                                      'Combined measures',
                                                      new_color=np.array([0.1, 0.8, 0.8]),
                                                      disc_rates=discount_stern)
efc_present = costben_disc.imp_meas_present['no measure']['efc']
efc_future = costben_disc.imp_meas_future['no measure']['efc']
efc_combined_measures = combined_costben_disc.imp_meas_future['Combined measures']['efc']

ax = plt.subplot(1, 1, 1)
efc_present.plot(axis=ax, color='blue', label='Present')
efc_future.plot(axis=ax, color='orange', label='Future, unadapted')
efc_combined_measures.plot(axis=ax, color='green', label='Future, adapted')
leg = ax.legend()


#%% Go over each method

#%%% Calc

''' 
    def calc(self, hazard, entity, haz_future=None, ent_future=None, future_year=None,
             risk_func=risk_aai_agg, imp_time_depen=None, save_imp=False, assign_centroids=True)
        
    Compute cost-benefit ratio for every measure provided current
    and, optionally, future conditions. Present and future measures need
    to have the same name. The measures costs need to be discounted by the user.
    If future entity provided, only the costs of the measures
    of the future and the discount rates of the present will be used.
    
    Parameters
    ----------
    hazard : climada.Hazard
    entity : climada.entity
    haz_future : climada.Hazard, optional
        hazard in the future (future year provided at ent_future)
    ent_future : Entity, optional
        entity in the future. Default is None
    future_year : int, optional
        future year to consider if no ent_future. Default is None
        provided. The benefits are added from the entity.exposures.ref_year until
        ent_future.exposures.ref_year, or until future_year if no ent_future given.
        Default: entity.exposures.ref_year+1
    risk_func : func optional
        function describing risk measure to use
        to compute the annual benefit from the Impact.
        Default: average annual impact (aggregated).
    imp_time_depen : float, optional
        parameter which represents time
        evolution of impact (super- or sublinear). If None: all years
        count the same when there is no future hazard nor entity and 1
        (linear annual change) when there is future hazard or entity.
        Default is None.
    save_imp : bool, optional
        Default: False
    assign_centroids : bool, optional
        indicates whether centroids are assigned to the self.exposures object.
        Centroids assignment is an expensive operation; set this to ``False`` to save
        computation time if the exposures from ``ent`` and ``ent_fut`` have already
        centroids assigned for the respective hazards.
        Default: True
    True if Impact of each measure is saved. Default is False.
'''



#%%% combine_measures

'''
Comment: Decide on how to combine measures but simply subtartving benefits per event should not do it 

'''



'''
 def combine_measures(self, in_meas_names, new_name, new_color, disc_rates,
                      imp_time_depen=None, risk_func=risk_aai_agg):
     
     """Compute cost-benefit of the combination of measures previously
     computed by calc with save_imp=True. The benefits of the
     measures per event are added. To combine with risk transfer options use
     apply_risk_transfer.

     Parameters
     ----------
     in_meas_names : list(str)
     list with names of measures to combine
     new_name :  str
         name to give to the new resulting measure
         new_color (np.array): color code RGB for new measure, e.g.
         np.array([0.1, 0.1, 0.1])
     disc_rates : DiscRates
         discount rates instance
     imp_time_depen : float, optional
         parameter which represents time
         evolution of impact (super- or sublinear). If None: all years
         count the same when there is no future hazard nor entity and 1
         (linear annual change) when there is future hazard or entity.
         Default is None.
     risk_func : func, optional
         function describing risk measure given
         an Impact. Default: average annual impact (aggregated).

     Returns
     -------
     climada.CostBenefit
     """

'''

# Investigate the difference adding two measures on top or combining the measures
# Compare aai and rp100

meas_1 = Measure(
    haz_type='TC',
    name='Measure A',
    color_rgb=np.array([0.8, 0.1, 0.1]),
    cost=5000000000,
    hazard_inten_imp=(1, -5),     # Decrease wind speeds by 5 m/s
    risk_transf_cover=0,
)

meas_2 = Measure(
    haz_type='TC',
    name='Measure B',
    color_rgb=np.array([0.1, 0.1, 0.8]),
    cost=220000000,
    paa_impact=(1, -0.10),   # 10% fewer assets affected
)

meas_12 = Measure(
    haz_type='TC',
    name='Measure AB',
    color_rgb=np.array([0.1, 0.1, 0.3]),
    cost=720000000,
    paa_impact=(1, -0.10),   # 10% fewer assets affected
    hazard_inten_imp=(1, -5),     # Decrease wind speeds by 5 m/s
    risk_transf_cover=0,
)

# Risk metrics
aai_ben_dict = {'Metric': ['aai'], 'Subtract': None, 'Compund': None, 'Combine': None}
rp100_ben_dict = {'Metric': ['rp100'], 'Subtract': None, 'Compund': None, 'Combine': None}



# aai and rp100
objects = (exp_present, impf_set, haz_present)
imp_no_measure = ImpactCalc(*objects).impact(save_mat=False)
aai_no_measure = imp_no_measure.aai_agg
rp100_no_measure = imp_no_measure.calc_freq_curve(return_per=np.array([100])).impact[0]

# Alternative 1 - Add the impact the metric
imp_meas_1= ImpactCalc(*meas_1.apply(*objects)).impact(save_mat=False)
imp_meas_2 = ImpactCalc(*meas_2.apply(*objects)).impact(save_mat=False)
aai_ben_dict['Subtract'] = [(aai_no_measure - imp_meas_1.aai_agg) + (aai_no_measure - imp_meas_2.aai_agg)]
rp100_ben_dict['Subtract'] = [(rp100_no_measure - imp_meas_2.calc_freq_curve(return_per=np.array([100])).impact[0]) + (rp100_no_measure - imp_meas_2.calc_freq_curve(return_per=np.array([100])).impact[0])]

# Alternative 2 - Do compounding
aai_ben_dict['Compund'] = [aai_no_measure - ImpactCalc(*meas_2.apply(*meas_1.apply(exp_present, impf_set, haz_present))).impact(save_mat=False).aai_agg]
rp100_ben_dict['Compund'] = [rp100_no_measure - ImpactCalc(*meas_2.apply(*meas_1.apply(exp_present, impf_set, haz_present))).impact(save_mat=False).calc_freq_curve(return_per=np.array([100])).impact[0]]


# Alternative 3 - Create a combination of all the measure attributes
aai_ben_dict['Combine'] = [aai_no_measure - ImpactCalc(*meas_12.apply(exp_present, impf_set, haz_present)).impact(save_mat=False).aai_agg]
rp100_ben_dict['Combine'] = [rp100_no_measure - ImpactCalc(*meas_12.apply(*objects)).impact(save_mat=False).calc_freq_curve(return_per=np.array([100])).impact[0]]

# Store results in data frame and print 
benefit_df = pd.DataFrame(aai_ben_dict)
benefit_df = pd.concat([benefit_df, pd.DataFrame(rp100_ben_dict)])

print(benefit_df)


#%%% apply_risk_transfer

'''
def apply_risk_transfer(self, meas_name, attachment, cover, disc_rates,
                        cost_fix=0, cost_factor=1, imp_time_depen=None,
                        risk_func=risk_aai_agg):
    """Applies risk transfer to given measure computed before with saved
    impact and compares it to when no measure is applied. Appended to
    dictionaries of measures.

    Parameters
    ----------
    meas_name : str
        name of measure where to apply risk transfer
    attachment : float
        risk transfer values attachment (deductible)
    cover : float
        risk transfer cover
    cost_fix : float
        fixed cost of implemented innsurance, e.g. transaction costs
    cost_factor : float, optional
        factor to which to multiply the insurance layer
        to compute its cost. Default is 1
    imp_time_depen : float, optional
        parameter which represents time
        evolution of impact (super- or sublinear). If None: all years
        count the same when there is no future hazard nor entity and 1
        (linear annual change) when there is future hazard or entity.
        Default is None.
    risk_func : func, optional
        function describing risk measure given
        an Impact. Default: average annual impact (aggregated).
    """
'''

#%%% remove_measure

'''
def remove_measure(self, meas_name):
    """Remove computed values of given measure

    Parameters
    ----------
    meas_name : str
        name of measure to remove
    """
    del self.color_rgb[meas_name]
    del self.benefit[meas_name]
    del self.cost_ben_ratio[meas_name]
    del self.imp_meas_future[meas_name]
    if self.imp_meas_present:
        del self.imp_meas_present[meas_name]

'''

#%%% plot_cost_benefit

'''
def plot_cost_benefit(self, cb_list=None, axis=None, **kwargs):
    """Plot cost-benefit graph. Call after calc().

    Parameters
    ----------
    cb_list : list(CostBenefit), optional
        if other CostBenefit
        provided, overlay them all. Used for uncertainty visualization.
    axis : matplotlib.axes._subplots.AxesSubplot, optional
        axis to use
    kwargs : optional
        arguments for Rectangle matplotlib, e.g. alpha=0.5
        (color is set by measures color attribute)

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
    """

'''


#%%% plot_event_view

'''
def plot_event_view(self, return_per=(10, 25, 100), axis=None, **kwargs):
    """Plot averted damages for return periods. Call after calc().

    Parameters
    ----------
    return_per : list, optional
        years to visualize. Default 10, 25, 100
    axis : matplotlib.axes._subplots.AxesSubplot, optional
        axis to use
    kwargs : optional
        arguments for bar matplotlib function, e.g. alpha=0.5
        (color is set by measures color attribute)

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
    """
    if not self.imp_meas_future:
        raise ValueError('Compute CostBenefit.calc() first')
    if not axis:
        _, axis = plt.subplots(1, 1)
    avert_rp = dict()
    for meas_name, meas_val in self.imp_meas_future.items():
        if meas_name == NO_MEASURE:
            continue
        interp_imp = np.interp(return_per, meas_val['efc'].return_per,
                               meas_val['efc'].impact)
        # check if measure over no measure or combined with another measure
        try:
            ref_meas = meas_name[meas_name.index('(') + 1:meas_name.index(')')]
        except ValueError:
            ref_meas = NO_MEASURE
        ref_imp = np.interp(return_per,
                            self.imp_meas_future[ref_meas]['efc'].return_per,
                            self.imp_meas_future[ref_meas]['efc'].impact)
        avert_rp[meas_name] = ref_imp - interp_imp

    m_names = list(self.cost_ben_ratio.keys())
    sort_cb = np.argsort(np.array([self.cost_ben_ratio[name] for name in m_names]))
    names_sort = [m_names[i] for i in sort_cb]
    color_sort = [self.color_rgb[name] for name in names_sort]
    ref_imp = np.interp(return_per, self.imp_meas_future[NO_MEASURE]['efc'].return_per,
                        self.imp_meas_future[NO_MEASURE]['efc'].impact)
    for rp_i, _ in enumerate(return_per):
        val_i = [avert_rp[name][rp_i] for name in names_sort]
        cum_effect = np.cumsum(np.array([0] + val_i))
        for (eff, color) in zip(cum_effect[::-1][:-1], color_sort[::-1]):
            axis.bar(rp_i + 1, eff, color=color, **kwargs)
        axis.bar(rp_i + 1, ref_imp[rp_i], edgecolor='k', fc=(1, 0, 0, 0), zorder=100)
    axis.set_xlabel('Return Period (%s)' % str(self.future_year))
    axis.set_ylabel('Impact (' + self.unit + ')')
    axis.set_xticks(np.arange(len(return_per)) + 1)
    axis.set_xticklabels([str(per) for per in return_per])
    return axis


'''




#%%% plot_waterfall

'''
def plot_waterfall_accumulated(self, hazard, entity, ent_future,
                               risk_func=risk_aai_agg, imp_time_depen=1,
                               axis=None, **kwargs):
    """Plot waterfall graph with accumulated values from present to future
    year. Call after calc() with save_imp=True. Provide same inputs as in calc.

    Parameters
    ----------
    hazard : climada.Hazard
    entity : climada.Entity
    ent_future : climada.Entity
        entity in the future
    risk_func : func, optional
        function describing risk measure given an Impact.
        Default: average annual impact (aggregated).
    imp_time_depen : float, optional
        parameter which represent time
        evolution of impact used in combine_measures. Default: 1 (linear).
    axis : matplotlib.axes._subplots.AxesSubplot, optional
        axis to use
    kwargs : optional
        arguments for bar matplotlib function, e.g. alpha=0.5

    Returns
    -------
        matplotlib.axes._subplots.AxesSubplot

'''

#%%% plot_arrow_averted


'''
def plot_arrow_averted(self, axis, in_meas_names=None, accumulate=False, combine=False,
                       risk_func=risk_aai_agg, disc_rates=None, imp_time_depen=1, **kwargs):
    """Plot waterfall graph with accumulated values from present to future
    year. Call after calc() with save_imp=True.

    Parameters
    ----------
    axis : matplotlib.axes._subplots.AxesSubplot
        axis from plot_waterfall
        or plot_waterfall_accumulated where arrow will be added to last bar
    in_meas_names : list(str), optional
        list with names of measures to
        represented total averted damage. Default: all measures
    accumulate : bool, optional)
        accumulated averted damage (True) or averted
        damage in future (False). Default: False
    combine : bool, optional
        use combine_measures to compute total averted
        damage (True) or just add benefits (False). Default: False
    risk_func : func, optional
        function describing risk measure given
        an Impact used in combine_measures. Default: average annual impact (aggregated).
    disc_rates : DiscRates, optional
        discount rates used in combine_measures
    imp_time_depen : float, optional
        parameter which represent time
        evolution of impact used in combine_measures. Default: 1 (linear).
    kwargs : optional
        arguments for bar matplotlib function, e.g. alpha=0.5
    """
    if not in_meas_names:
        in_meas_names = list(self.benefit.keys())
    bars = [rect for rect in axis.get_children() if isinstance(rect, Rectangle)]

    if accumulate:
        tot_benefit = np.array([self.benefit[meas] for meas in in_meas_names]).sum()
        norm_fact = self.tot_climate_risk / bars[3].get_height()
    else:
        tot_benefit = np.array([risk_func(self.imp_meas_future[NO_MEASURE]['impact']) -
                                risk_func(self.imp_meas_future[meas]['impact'])
                                for meas in in_meas_names]).sum()
        norm_fact = (risk_func(self.imp_meas_future['no measure']['impact'])
                     / bars[3].get_height())
    if combine:
        try:
            LOGGER.info('Combining measures %s', in_meas_names)
            all_meas = self.combine_measures(in_meas_names, 'combine',
                                             colors.to_rgba('black'), disc_rates,
                                             imp_time_depen, risk_func)
        except KeyError:
            LOGGER.warning('Use calc() with save_imp=True to get a more accurate '
                           'approximation of total averted damage,')
        if accumulate:
            tot_benefit = all_meas.benefit['combine']
        else:
            tot_benefit = risk_func(all_meas.imp_meas_future[NO_MEASURE]['impact']) - \
                risk_func(all_meas.imp_meas_future['combine']['impact'])

    self._plot_averted_arrow(axis, bars[3], tot_benefit, bars[3].get_height() * norm_fact,
                             norm_fact, **kwargs)
'''

#%%% plot_waterfall_accumulated


'''
@staticmethod
def plot_waterfall(hazard, entity, haz_future, ent_future,
                   risk_func=risk_aai_agg, axis=None, **kwargs):
    """Plot waterfall graph at future with given risk metric. Can be called
    before and after calc().

    Parameters
    ----------
    hazard : climada.Hazard
    entity : climada.Entity
    haz_future : Hazard
        hazard in the future (future year provided at ent_future).
        ``haz_future`` is expected to have the same centroids as ``hazard``.
    ent_future : climada.Entity
        entity in the future
    risk_func : func, optional
        function describing risk measure given
        an Impact. Default: average annual impact (aggregated).
    axis : matplotlib.axes._subplots.AxesSubplot, optional
        axis to use
    kwargs : optional
        arguments for bar matplotlib function, e.g. alpha=0.5

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
    """
    if ent_future.exposures.ref_year == entity.exposures.ref_year:
        raise ValueError('Same reference years for future and present entities.')
    present_year = entity.exposures.ref_year
    future_year = ent_future.exposures.ref_year

    imp = ImpactCalc(entity.exposures, entity.impact_funcs, hazard)\
          .impact(assign_centroids=hazard.centr_exp_col not in entity.exposures.gdf)
    curr_risk = risk_func(imp)

    imp = ImpactCalc(ent_future.exposures, ent_future.impact_funcs, haz_future)\
          .impact(assign_centroids=hazard.centr_exp_col not in ent_future.exposures.gdf)
    fut_risk = risk_func(imp)

    if not axis:
        _, axis = plt.subplots(1, 1)
    norm_fact, norm_name = _norm_values(curr_risk)

    # current situation
    LOGGER.info('Risk at {:d}: {:.3e}'.format(present_year, curr_risk))

    # changing future
    # socio-economic dev
    imp = ImpactCalc(ent_future.exposures, ent_future.impact_funcs, hazard)\
          .impact(assign_centroids=False)
    risk_dev = risk_func(imp)
    LOGGER.info('Risk with development at {:d}: {:.3e}'.format(future_year, risk_dev))

    # socioecon + cc
    LOGGER.info('Risk with development and climate change at {:d}: {:.3e}'.
                format(future_year, fut_risk))

    axis.bar(1, curr_risk / norm_fact, **kwargs)
    axis.text(1, curr_risk / norm_fact, str(int(round(curr_risk / norm_fact))),
              horizontalalignment='center', verticalalignment='bottom',
              fontsize=12, color='k')
    axis.bar(2, height=(risk_dev - curr_risk) / norm_fact,
             bottom=curr_risk / norm_fact, **kwargs)
    axis.text(2, curr_risk / norm_fact + (risk_dev - curr_risk) / norm_fact / 2,
              str(int(round((risk_dev - curr_risk) / norm_fact))),
              horizontalalignment='center', verticalalignment='center', fontsize=12, color='k')
    axis.bar(3, height=(fut_risk - risk_dev) / norm_fact,
             bottom=risk_dev / norm_fact, **kwargs)
    axis.text(3, risk_dev / norm_fact + (fut_risk - risk_dev) / norm_fact / 2,
              str(int(round((fut_risk - risk_dev) / norm_fact))),
              horizontalalignment='center', verticalalignment='center', fontsize=12,
              color='k')
    axis.bar(4, height=fut_risk / norm_fact, **kwargs)
    axis.text(4, fut_risk / norm_fact, str(int(round(fut_risk / norm_fact))),
              horizontalalignment='center', verticalalignment='bottom',
              fontsize=12, color='k')

    axis.set_xticks(np.arange(4) + 1)
    axis.set_xticklabels(['Risk ' + str(present_year),
                          'Economic \ndevelopment',
                          'Climate \nchange',
                          'Risk ' + str(future_year)])
    axis.set_ylabel('Impact (' + imp.unit + ' ' + norm_name + ')')
    axis.set_title('Risk at {:d} and {:d}'.format(present_year, future_year))
    return axis

'''