import os
import orca
import yaml
import json
import plotly
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from collections import OrderedDict

# -----------------------------------------------------------------------------------------
# GENERATES OUTPUT INDICATORS AND CHARTS
# -----------------------------------------------------------------------------------------
def gen_all_indicators(indicator_params, year):
    for key in indicator_params.keys():
        vars = indicator_params[key]
        for geo in vars.keys():
            if vars[geo] is not None:
                print('Exporting ', geo, 'csv for ', key)
                tbl = geo.replace('blockgroups', 'block_groups').replace('county', 'counties')
                vars[geo] = [var.replace('_nc', '_noclip') for var in vars[geo]]
                indicators = orca.get_table(tbl).to_frame(vars[geo])
                indicators = indicators.reindex(sorted(indicators.columns), axis=1)
                if geo == 'blockgroups':
                    indicators.index.name = 'blockgroup_id'
                rename_vars = {var: var.replace('_noclip', '_nc') for var in indicators.columns}
                indicators = indicators.rename(columns=rename_vars)
                region_code = orca.get_injectable('region_code')
                indicators.to_csv('./runs/%s_%s_%s_indicators_%s.csv' % (region_code, geo.strip('s'), key.strip('s'), year))


def gen_all_charts(chart_params, base_year, forecast_year, data, geo_large):
    vdict = orca.get_injectable('var_dict')

    # Charts for initial run (base year charts and calibration/validation)
    if orca.get_injectable('initial_run') == True:
        base_chart_vars = chart_params['variables']['base_run_charts']
        for aggtype in ['sum', 'mean']:
            for chart_type in base_chart_vars[aggtype].keys():
                vars = base_chart_vars[aggtype][chart_type]
                eval('gen_%s(data, base_year, vars, aggtype, geo_large, vdict, initial_run=True)' % (chart_type))
        calibration_scatter_vars = base_chart_vars['calibration_scatters']
        gen_scatters(data, forecast_year, calibration_scatter_vars, 'sum', geo_large, vdict, initial_run = True, validation=True)

    # Charts to summarize forecasting results
    for aggtype in ['sum', 'mean']:
        for chart_type in chart_params['variables']['forecasting_charts'][aggtype].keys():
            vars = chart_params['variables']['forecasting_charts'][aggtype][chart_type]
            print('Running gen_%s function' % (chart_type))
            eval('gen_%s(data, forecast_year, vars, aggtype, geo_large, vdict)' % (chart_type))


def prepare_chart_data(cfg, year):
    geo_small = cfg['output_charts']['geography_small']
    geo_large = cfg['output_charts']['geography_large']
    all_vars = []
    for key in cfg['output_indicators'].keys():
        vars = cfg['output_indicators'][key]['tracts']
        all_vars += [var.replace('_noclip', '_nc') for var in vars]
    all_vars = list(set(all_vars))
    data = {}
    region = orca.get_injectable('region_code')
    base_year = orca.get_injectable('base_year')
    for i in range(base_year, year+1):
        filename = 'runs/%s_%s_download_indicators_%s.csv' % (region, geo_small.replace('_id', ''), i)
        data_year = pd.read_csv(filename, dtype={geo_small: object}).set_index(geo_small)[all_vars]
        data_year['county_id'] = data_year.index.str.slice(0,5)
        data[i] = data_year
    return data, geo_small, geo_large


def aggregate_data(data, agg_type, geo):
    if agg_type == 'mean':
        data = data.groupby(geo).mean().reset_index()
    else:
        data = data.groupby(geo).sum().reset_index()
    return data


def gen_pie_charts(all_data, year, vars, agg_type, geo, vdict, initial_run=False):
    data = {yr_key: all_data[yr_key] for yr_key in all_data.keys() if yr_key>2010}
    for var in vars:
        pie_chart = px.pie()
        for year in data.keys():
            data_year = aggregate_data(data[year], agg_type, geo)
            trace = go.Pie(values=data_year[var], labels=data_year[geo])
            pie_chart.add_trace(trace)
        pie_chart.data[0].visible = True
        steps = []
        i = 0
        for year in data.keys():
            step = dict(method="restyle", args=["visible", [False] * len(pie_chart.data)], label='{}'.format(year))
            step["args"][1][:i + 1] = [True] * (i + 1)
            steps.append(step)
            i += 1
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Year: "},
            pad={"t": 10},
            steps=steps)]
        pie_chart.update_layout(sliders=sliders)
        path = './runs/forecast_year_%s_by_%s_pie_chart.json' % (var, geo)
        if initial_run:
            path = path.replace('forecast', 'base')
        with open(path, 'w') as outfile:
            json.dump(pie_chart.to_json(), outfile)


def gen_scatters(data, year, vars, aggtype, geo, vdict, initial_run=False, validation=False):
    data = data[year].reset_index()
    for var1 in vars.keys():
        for var2 in vars[var1]:
            titley = vdict[var1].replace('Calibration var: ', '').replace('Validation var: ', '').capitalize()
            titlex = vdict[var2].replace('Calibration var: ', '').replace('Validation var: ', '').capitalize()
            labels = {var1: titley, var2: titlex, 'tract_id': 'tract', 'county_id': 'county'}
            agg_data1 = aggregate_data(data, aggtype, geo)[[var1 , geo]]
            agg_data2 = aggregate_data(data, aggtype, geo)[[var2, geo]]
            agg_data = pd.merge(agg_data1, agg_data2, on=geo, how='left')
            for data_source in ['data', 'agg_data']:
                df = eval(data_source)
                hover_data = ['tract_id'] if data_source == 'data' else [geo]
                scatter = px.scatter(df, x=var2, y=var1, labels=labels, hover_data=hover_data)
                corr = round(df.corr().loc[var2][var1], 2)
                scatter.add_annotation(x=df[var2].max()*0.1, y=df[var1].max()*0.95, text='corr:' + str(corr), showarrow=False)
                path = './runs/forecast_year_%s_vs_%s.json' % (var1, var2)
                if 'agg_' in data_source:
                    path = path.replace('.json', '%s.json' % (geo.replace('_id', '')))
                if initial_run:
                    path = path.replace('forecast', 'base')
                if validation:
                    path = path.replace('base_year', 'calibration')
                with open(path, 'w') as outfile:
                    json.dump(scatter.to_json(), outfile)
                if ('prop_sim_' in var1) and (orca.get_injectable('local_simulation') == True):
                    title = 'Simulated vs observed proportion of growth by tract during microsimulation'
                    path = './runs/%s_tract_%s_vs_%s.png' % (orca.get_injectable('region_code'), var1, var2.replace('_nc', '_noclip'))
                    if 'agg_' in data_source:
                        title = title.replace('tract', geo.replace('_id', ''))
                        path = path.replace('tract', geo.replace('_id', ''))
                    scatter = px.scatter(df, x=var2, y=var1, labels={var2: 'observed proportion', var1: 'simulated proportion'}, title=title)
                    scatter.write_image(path)


def gen_bar_charts(data, year, vars, agg_type, geo, vdict, initial_run=False):
    data = data[year].copy()
    data = aggregate_data(data, agg_type, geo)
    for var in vars:
        if 'prop_' in var and agg_type == 'sum':
            agent = var.split('_')[1]
            category = var.split('_')[2]
            data[var] = data['%s_%s' % (agent, category)] / data['total_%s' % agent]
        labels = {var: vdict[var], geo: geo.replace('_id', ' ')}
        bar_chart = px.bar(data, x=var, y=geo, labels=labels)
        path = './runs/forecast_year_%s_by_%s.json' % (var, geo)
        if initial_run:
            path = path.replace('forecast', 'base')
        with open(path, 'w') as outfile:
            json.dump(bar_chart.to_json(), outfile)


def gen_histograms(data, year, vars, agg_type, geo, vdict, initial_run=False):
    data = data[year]
    for var in vars:
        titlex = vdict[var]
        if vdict[var].split(' ')[0] == 'Log':
            titlex = 'log of ' + vdict[var].lower()
        hist = px.histogram(data, x=var, labels= {var: titlex})
        path = './runs/forecast_year_%s_histogram.json' % var
        if initial_run:
            path = path.replace('forecast', 'base')
        with open(path, 'w') as outfile:
            json.dump(hist.to_json(), outfile)


def gen_bar_charts_n_largest(data, year, vars, agg_type, geo, vdict, n=10, initial_run=False):
    data = data[year]
    for var in vars:
        data = aggregate_data(data, agg_type, geo)
        max_data = data.nlargest(n, var).reset_index()
        labels = {var: '%s of %s' % (agg_type, vdict[var]), geo: geo.replace('_id', ' ')}
        bar_chart = px.bar(max_data, x=var, y=geo, labels=labels)
        path = './runs/forecast_year_%s_%ss_with_max_%s.json'% (n, geo, var)
        if initial_run:
            path = path.replace('forecast', 'base')
        with open(path, 'w') as outfile:
            json.dump(bar_chart.to_json(), outfile)


def gen_bar_charts_totals(data, year, vars, agg_type, geo, vdict, initial_run=False):
    if len(vars)>0:
        data = aggregate_data(data[year], agg_type, geo)
        data = pd.DataFrame(data.set_index(geo)[vars].stack(), columns=['value']).reset_index()
        data = data.rename(columns={'level_1': 'agent'})
        bar_chart = px.bar(data, x='agent', y='value', color=geo,  barmode="group")
        path = './runs/forecast_year_total_agents_by_%s.json' % (geo)
        if initial_run:
            path = path.replace('forecast', 'base')
        with open(path, 'w') as outfile:
            json.dump(bar_chart.to_json(), outfile)


def gen_bar_charts_proportions(data, year, vars, agg_type, geo, vdict, initial_run=False):
    data = data[year]
    for agent in vars.keys():
        agents = orca.get_table(agent)
        for var in vars[agent]:
            df = pd.DataFrame()
            agents_by_cat = agents[var].value_counts()
            agent = agent.replace('households', 'hh').replace('residential_', '')
            for cat in agents_by_cat.index.values:
                cat = str(cat)
                if (agent == 'hh') and (var == 'tenure'):
                    cat = cat.replace('1', 'own').replace('2', 'rent')
                new_var = var
                for text in ['agg_sector', 'segment', 'building_type_id', 'building_type', 'tenure']:
                    new_var = new_var.replace(text, '')
                new_var = "%s_%s_%s" % (agent, new_var, str(cat))
                new_var = new_var.replace('__', '_')
                df_var = aggregate_data(data[[geo, new_var]], agg_type, geo)
                df_var = df_var.rename(columns={new_var: agent})
                df_var[var] = cat
                df = df.append(df_var)
            df_agg = df.groupby(geo)[agent].sum().reset_index().rename(columns={agent: agent + '_' + geo})
            df = df.merge(df_agg, on=geo, how='left')
            df['prop'] = df[agent]/df[agent + '_' + geo]*100
            labels = {'prop': 'Proportion of '+ agent, geo: geo.replace('_id', ' '), var: var.replace('_', ' ')}
            bar_chart = px.bar(df, x='prop', y=geo, color=var, labels=labels)
            bar_chart.update_layout(yaxis=dict(type='category'))
            agent = agent.replace('hh', 'households').replace('units', 'residential_units')
            path = './runs/forecast_year_prop_%s_by_%s_by_%s.json' % (agent, var, geo)
            if initial_run:
                path = path.replace('forecast', 'base')
            with open(path, 'w') as outfile:
                json.dump(bar_chart.to_json(), outfile)


def gen_custom_barchart(table, var):
    df = orca.get_table(table).to_frame(['block_id', var]).\
        groupby(var).count().reset_index()
    df.rename(columns={'block_id': 'count_'+table}, inplace=True)
    bar_chart = px.bar(df, x='count_' + table, y=var)
    with open('./runs/%s_by_%s.json' % (table, var), 'w') as outfile:
        json.dump(bar_chart.to_json(), outfile)

# -----------------------------------------------------------------------------------------
# EXPORTS METRICS FOR CALIBRATION ROUTINE
# -----------------------------------------------------------------------------------------

def gen_calibration_metrics(tracts):
    tract_cols = [col for col in tracts.columns if ('_obs_growth' in col or '_sim_growth' in col)]
    tract_calibration_data = tracts.to_frame(tract_cols)
    tract_calibration_data['county_id'] = tract_calibration_data.index.str.slice(0, 5)
    county_calibration_data = tract_calibration_data.groupby('county_id').sum()
    metrics = pd.DataFrame(index=['ELCM', 'HLCM', 'RDPLCM'])
    for geo in ['county', 'tract']:
        geo_corr = eval(geo + '_calibration_data').corr()
        corrs = pd.DataFrame([['ELCM', geo_corr.loc['jobs_prop_sim_growth_10_17']['jobs_prop_obs_growth_10_17']],
                              ['HLCM', geo_corr.loc['hh_prop_sim_growth_13_18']['hh_prop_obs_growth_13_18']],
                              ['RDPLCM',geo_corr.loc['units_prop_sim_growth_13_18']['units_prop_obs_growth_13_18']]],
                             columns=['model', geo + '_sim_corr'])
        corrs_noclip = pd.DataFrame(
            [['ELCM', geo_corr.loc['jobs_prop_sim_growth_10_17']['jobs_prop_obs_growth_10_17_noclip']],
             ['HLCM', geo_corr.loc['hh_prop_sim_growth_13_18']['hh_prop_obs_growth_13_18_noclip']],
             ['RDPLCM', geo_corr.loc['units_prop_sim_growth_13_18']['units_prop_obs_growth_13_18_noclip']]],
            columns=['model', geo + '_sim_corr_noclip'])
        print('-------------------------------------')
        print(geo, ' correlations:')
        print(corrs)
        print(geo, ' correlations including negatives:')
        print(corrs_noclip)
        metrics = metrics.join(corrs.set_index('model'))
        metrics = metrics.join(corrs_noclip.set_index('model'))
    metrics.index.name = 'submodel'
    metrics.to_csv('runs/' + orca.get_injectable('region_code') + '_metrics.csv')
    print('-------------------------------------')


# -----------------------------------------------------------------------------------------
# CREATES DICTIONARY WITH METADATA FOR UI
# -----------------------------------------------------------------------------------------


def create_variable_dictionary():
    with open("configs/output_parameters.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        orca.add_injectable('output_parameters', cfg)
    base_dict = cfg['variable_definitions']['base_dict']
    custom_dict = cfg['variable_definitions']['custom_dict']
    year_vars = cfg['variable_definitions']['year_vars']
    prop_vars = cfg['variable_definitions']['prop_vars']
    updated_dict = adds_calibration_dict(year_vars, base_dict)
    updated_dict = adds_categories_dict(prop_vars, updated_dict)
    updated_dict = adds_derived_vars_dict(updated_dict)
    updated_dict = adds_undefined_vars_dict(updated_dict, cfg)
    full_dict = {'var_dict': updated_dict}
    full_dict['custom_var_dict'] = custom_dict
    return full_dict


def adds_calibration_dict(year_vars, dict):
    base_vars = list(year_vars.keys())
    for base_var in base_vars:
        prop_var = base_var.replace('_sim', '_prop_sim')
        year_vars[prop_var] = year_vars[base_var]
        dict['other'][prop_var] = dict['other'][base_var].replace('Total simulated', 'Percentage of simulated')
    for var in year_vars.keys():
        for year in year_vars[var]:
            sim_var = var.replace('year', year)
            var_type = 'Calibration' if year in ['10_17', '13_18'] else 'Validation'
            var_def = dict['other'][var].replace('Percentage of simulated', 'simulated % of')
            var_def = var_def.replace('Total simulated', 'simulated')
            var_def += ' between 20%s and 20%s' % (year.split('_')[0], year.split('_')[1])
            dict['other'][sim_var] = '%s var: %s' % (var_type, var_def)
            obs_var = sim_var.replace('_sim_', '_obs_')
            dict['other'][obs_var] = dict['other'][sim_var].replace('simulated', 'observed')
            unclipped_var = obs_var + '_nc'
            dict['other'][unclipped_var] = dict['other'][obs_var].replace('growth', 'change')
    return dict


def adds_categories_dict(prop_vars, dict):
    prop = {}
    simple = {}
    for agent in prop_vars:
        vars = prop_vars[agent]
        agents = orca.get_table(agent)
        for var in vars:
            agents_by_cat = agents[var].value_counts()
            cats_to_measure = agents_by_cat[agents_by_cat > 500].index.values.astype('str')
            for cat in cats_to_measure:
                base_desc, prop_desc = gen_category_descriptions(agent, var, cat)
                new_var = format_category_var(agent, var, cat)
                simple[new_var] = base_desc
                prop[new_var] = prop_desc
    dict['other'].update(simple)
    dict['prop'] = prop
    return dict


def gen_category_descriptions(agent, var, cat):
    if (agent == 'households') and (var == 'tenure'):
        cat = cat.replace('1', 'own').replace('2', 'rent')
    desc = var + ' '
    for text in ['building_type_id ', 'tenure ', 'year_']:
        desc = desc.replace(text, '')
    if (agent == 'residential_units') and (var == 'year_built'):
        desc = desc.replace('1930', 'before 1930').replace('2010', 'after 2000')
    base_desc = 'Total ' + agent + ' ' + desc.replace('_', ' ') + str(cat)
    prop_desc = 'Proportion of ' + agent + ' ' + desc.replace('_', ' ') + str(cat)
    return base_desc.capitalize(), prop_desc


def format_category_var(agent, var, cat):
    agent = agent.replace('households', 'hh').replace('residential_', '')
    if (agent == 'hh') and (var == 'tenure'):
        cat = cat.replace('1', 'own').replace('2', 'rent')
    for text in ['agg_sector', 'tenure', 'building_type_id']:
        var = var.replace(text, '')
    if var == 'income_segment':
        new_var = "%s_%s_%s" % (var, str(cat), agent)
    else:
        new_var = "%s_%s_%s" % (agent, var, str(cat))
    new_var = new_var.replace('__', '_')
    return new_var


def adds_derived_vars_dict(dict):
    new_dict = {}
    derived_vars = {key:dict[key].keys() for key in dict.keys()}

    for agg in ['total', 'sum', 'mean', 'prop', 'other']:
        for var in dict[agg]:
            if agg != 'other':
                new_var = agg + '_' + var
            else:
                new_var = var
            new_dict[new_var] = dict[agg][var]
            new_dict['bg_' + new_var] = dict[agg][var] + ' (block group level)'
            new_dict['zones_' + new_var] = dict[agg][var] + ' (zones level)'

    thresholds = orca.get_injectable('impedance_thresholds')
    units = orca.get_injectable('impedance_units')
    methods = orca.get_injectable('skim_input_columns')
    zones_table = orca.get_injectable('zones_table')
    for dist in thresholds:
        for method in methods:
            for agg in ['total', 'sum', 'mean', 'prop', 'other']:
                for var in derived_vars[agg]:
                    new_var = (agg + '_').replace('other_', '') + var + '_ave_' + str(dist) + '_' + units + '_' + method
                    for text in ['_segment']:
                        new_var = new_var.replace(text, '')
                    desc = 'Skim var: Average of ' + dict[agg][var].lower() + ' within ' + str(dist) + units
                    desc = desc.replace('mean ', '').replace('total ', '')
                    new_dict[new_var] = desc
                    new_var = '%s_%s' % (zones_table.replace('block_groups', 'bg'), new_var)
                    new_dict[new_var] = '%s (%s level)' % (desc, zones_table.replace('_', ' '))

            for agg in ['total', 'sum', 'other']:
                for var in derived_vars[agg]:
                    new_var = (agg + '_').replace('other_', '') + var + '_sum_' + str(dist) + '_' + units + '_' + method
                    for text in ['_segment']:
                        new_var = new_var.replace(text, '')
                    desc = 'Skim var: Sum of ' + dict[agg][var].lower() + ' within ' + str(dist) + units
                    desc = desc.replace('total ', '')
                    new_dict[new_var] = desc
                    new_var = '%s_%s' % (zones_table.replace('block_groups', 'bg'), new_var)
                    new_dict[new_var] = '%s (%s level)' % (desc, zones_table.replace('_', ' '))

    var_list = list(new_dict.keys())
    for var in var_list:
        new_dict['ln_' + var] = 'Natural logarithm of ' + new_dict[var].lower()

    distances = range(400, 5000, 800)
    units = 'm'
    decay_types = ['linear', 'flat']
    for dist in distances:
        for decay in decay_types:
            for agg in ['mean', 'prop', 'other']:
                for var in derived_vars[agg]:
                    if '_growth_' not in var:
                        new_var = var + '_ave_' + str(dist) + '_' + decay[0]
                        if agg == 'prop':
                            new_var = 'prop_' + new_var
                        if 'income_segment' in new_var:
                            new_var = new_var.replace('_hh', '')
                        for text in ['_segment']:
                            new_var = new_var.replace(text, '')
                        new_var = new_var.replace('before_1930', 'old')
                        new_var = new_var.replace('after_2000', 'new')
                        desc = 'Pandana var: Average ' + dict[agg][var].lower() + ' within ' + str(dist) + units
                        desc = desc.replace('mean ', '').replace('total ', '')
                        new_dict[new_var] = desc
                        new_dict['ln_' + new_var] = desc.replace('Average', 'Log of average')

            for agg in ['total', 'sum', 'other']:
                for var in derived_vars[agg]:
                    if '_growth_' not in var:
                        new_var = var + '_sum_' + str(dist) + '_' + decay[0]
                        for text in ['_segment']:
                            new_var = new_var.replace(text, '')
                        desc = 'Pandana var: Sum of ' + dict[agg][var].lower() + ' within ' + str(dist) + units
                        desc = desc.replace('total ', '')
                        new_dict[new_var] = desc
                        new_dict['ln_' + new_var] = desc.replace('Sum of', 'Log of sum of')

    var_list = list(new_dict.keys())
    for var in var_list:
        new_dict['st_' + var] = 'standarized ' + new_dict[var].lower()
    return new_dict


def adds_undefined_vars_dict(dict, cfg):
    all_vars = []
    for key in cfg['output_indicators'].keys():
        vars = cfg['output_indicators'][key]
        for tbl in vars.keys():
            all_vars += vars[tbl]
    all_vars = list(set(all_vars))
    undefined_vars = [var for var in all_vars if var not in dict.keys()]
    for var in undefined_vars:
        dict[var] = var.replace('_', ' ').capitalize()
    return dict


# -----------------------------------------------------------------------------------------
# EXPORTS METADATA FOR UI
# -----------------------------------------------------------------------------------------

def export_indicator_definitions():
    print('Exporting indicator definitions...')
    # Gets variable definitions from var_dict
    full_dict = create_variable_dictionary()
    var_dict = full_dict['var_dict']
    custom_d = full_dict['custom_var_dict']
    output_params = orca.get_injectable('output_parameters')

    # Creates metadata for output layers
    data = {}
    spatial_output = {}
    layer_vars = output_params['output_indicators']['layers']
    for geo_type in layer_vars:
        desc = {}
        variables = layer_vars[geo_type]
        for var in variables:
            geo_type = geo_type.strip('s')
            desc[var] = {'name': var_dict[var]}
        csv = orca.get_injectable('region_code') + '_' + geo_type + '_layer_indicators'
        spatial_output[geo_type.replace('block_group', 'blockgroup')] = {'root_csv_name': csv, 'var_display': desc}
    data['spatial_output'] = OrderedDict(spatial_output)

    # Creates metadata for downloads
    download_output = {}
    default_downloads = {}
    if 'downloads' in output_params['output_indicators'].keys():
        for geo_type in output_params['output_indicators']['downloads']:
            display_name = geo_type.capitalize()
            geo_type = geo_type.strip('s')
            csv = orca.get_injectable('region_code') + '_' + geo_type + '_download_indicators'
            downloads_geo = {'display_name': display_name, 'root_csv_name': csv}
            default_downloads[geo_type] = downloads_geo
        download_output['default_downloads'] = default_downloads
        data['download_output'] = download_output

    # Creates metadata for charts
    geo_large = output_params['output_charts']['geography_large']
    geo_small = output_params['output_charts']['geography_small']
    chart_output = []
    periods = ['base_run', 'forecasting'] if orca.get_injectable('initial_run') == True else ['forecasting']
    for period in periods:
        year_name = 'base' if period == 'base_run' else 'forecast'
        charts_period = output_params['output_charts']['variables'][period + '_charts']
        for var_type in ['sum', 'mean']:
            scatter_vars = charts_period[var_type]['scatters']
            if scatter_vars is not None:
                for var1 in scatter_vars.keys():
                    for var2 in scatter_vars[var1]:
                        tract_scatter = {'file_name': ('%s_year_%s_vs_%s.json' % (year_name, var1, var2))}
                        county_scatter = {'file_name': ('%s_year_%s_vs_%s_county.json' % (year_name, var1, var2))}
                        varname1 = var_dict[var1].replace(': ', ' ')
                        varname2 = var_dict[var2].replace(': ', ' ')
                        tract_scatter['title'] = '%s year: Tract level %s vs. %s' % (year_name.capitalize(), varname1, varname2)
                        county_scatter['title'] = '%s year: County level %s vs. %s' % (year_name.capitalize(), varname1, varname2)
                        chart_output += [tract_scatter]
                        chart_output += [county_scatter]
            pie_chart_vars = charts_period[var_type]['pie_charts']
            for var in pie_chart_vars:
                piechart = {'file_name': ('%s_year_%s_by_%s_pie_chart.json' % (year_name, var, geo_large))}
                varname = var_dict[var].replace(': ', ' ')
                piechart['title'] = year_name.capitalize() + ' year: ' + varname + ' by ' + geo_large.replace('_id', ' id')
                chart_output += [piechart]
            bar_chart_vars = charts_period[var_type]['bar_charts']
            for var in bar_chart_vars:
                barchart = {'file_name': ('%s_year_%s_by_%s.json' % (year_name, var, geo_large))}
                varname = var_dict[var].replace(': ', ' ')
                barchart['title'] = year_name.capitalize() + ' year: ' + varname + ' by ' + geo_large.replace('_id', ' id')
                chart_output += [barchart]
            histogram_vars = charts_period[var_type]['histograms']
            for var in histogram_vars:
                histogram = {'file_name': '%s_year_%s_histogram.json' % (year_name, var)}
                varname = (var_dict[var]).strip('Log of ').replace(': ', ' ')
                histogram['title'] = year_name.capitalize() + ' year: Count of ' + geo_small.replace('_id', 's') + ' by ' + varname.lower()
                chart_output += [histogram]
            n_largest_vars = charts_period[var_type]['bar_charts_n_largest']
            for var in n_largest_vars:
                n_largest_chart = {'file_name': '%s_year_%s_%ss_with_max_%s.json' % (year_name, 10, geo_small, var)}
                varname = var_dict[var].replace(': ', ' ')
                n_largest_chart['title'] = year_name.capitalize() + ' year: Ten ' + geo_small.replace('_id', 's') + ' with highest ' + varname
                chart_output += [n_largest_chart]
            proportion_vars = charts_period[var_type]['bar_charts_proportions']
            for agent in proportion_vars.keys():
                for var in proportion_vars[agent]:
                    proportion_chart = {'file_name': '%s_year_prop_%s_by_%s_by_%s.json' % (year_name, agent, var, geo_large)}
                    proportion_chart['title'] = '%s year: Proportion of %s by %s by %s' % (year_name.capitalize(), agent, var, geo_large)
                    chart_output += [proportion_chart]
        if len(charts_period['sum']['bar_charts_totals']) > 0:
            totals_chart = {'file_name': '%s_year_total_agents_by_%s.json' % (year_name, geo_large)}
            totals_chart['title'] = '%s year: Total agents by %s' % (year_name.capitalize(), geo_large)
            chart_output += [totals_chart]


    if orca.get_injectable('initial_run') == True:
        base_run_calibration_scatter_vars = output_params['output_charts']['variables']['base_run_charts']['calibration_scatters']
        for var1 in base_run_calibration_scatter_vars.keys():
            for var2 in base_run_calibration_scatter_vars[var1]:
                tract_scatter = {'file_name': ('calibration_%s_vs_%s.json' % (var1, var2))}
                county_scatter = {'file_name': ('calibration_%s_vs_%s_county.json' % (var1, var2))}
                var_type = 'Calibration' if '10_17' in var2 or '13_18' in var2 else 'Validation'
                var_def = var_dict[var2].replace('Calibration var: ', '').replace('Validation var: ', '')
                tract_scatter['title'] = '%s: Simulated vs. %s' % (var_type, var_def.replace('between', 'by tract between'))
                county_scatter['title'] = tract_scatter['title'].replace('tract', 'county')
                chart_output += [tract_scatter]
                chart_output += [county_scatter]


    # Creates metadata for custom charts
    custom_v = output_params['output_charts']['variables']['forecasting_charts']['custom_charts']
    for table in custom_v:
        if custom_v[table] is not None:
            for var in custom_v[table]:
                custom_chart = {'file_name': '%s_by_%s.json' % (table, var)}
                key = table + '_' + var
                try:
                    data_name = custom_d[key]['data_name']
                    agg_name = custom_d[key]['aggregation_name']
                except Exception:
                    data_name = 'Total ' + table.replace('_', ' ')
                    agg_name = var.replace('_', ' ')
                custom_chart['title'] = data_name + ' by ' + agg_name
                chart_output += custom_chart
    chart_diagnostics = {'chart_library': 'plotly', 'chart_library_version': plotly.__version__, 'python_version': 3.8}
    complete_chart_output = []
    from copy import deepcopy
    for chart in chart_output:
        complete_chart = chart.copy()
        complete_chart['chart_diagnostics'] = chart_diagnostics
        complete_chart_output += [OrderedDict(deepcopy(complete_chart))]
    data['chart_output'] = complete_chart_output

    # Exports indicator and chart metadata to .yaml file
    data = OrderedDict(data)
    represent_dict_order = lambda self, data: \
        self.represent_mapping('tag:yaml.org,2002:map', data.items())
    yaml.add_representer(OrderedDict, represent_dict_order)
    with open('./runs/output_indicator_definitions.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, width = 1000)
    print('Indicator definitions exported.')
    orca.add_injectable('var_dict', var_dict)
    orca.add_injectable('custom_dict', custom_d)
