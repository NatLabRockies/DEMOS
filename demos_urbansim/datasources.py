import glob
import os
from itertools import product

import numpy as np
import openmatrix as omx
import orca
import pandas as pd
import yaml
from google.cloud import storage
from urbansim_templates import modelmanager as mm
from urbansim_templates.data import LoadTable
from urbansim_templates.models import LargeMultinomialLogitStep, OLSRegressionStep

print("importing datasources")

# -----------------------------------------------------------------------------------------
# UC SIMULATIONS: ADDS SPECIAL SCENARIO INJECTABLES FROM NOTES
# -----------------------------------------------------------------------------------------
scenario_data = glob.glob("./data/scenario_data*")
if len(scenario_data) > 0:
    with open(scenario_data[0]) as f:
        scenario = yaml.load(f, Loader=yaml.FullLoader)
    if scenario["notes"] is not None and scenario["notes"] != "":
        print("Extracting relevant information from scenario notes")
        settings = eval(scenario["notes"])["settings"]
        if "calibrated_folder" in settings.keys():
            orca.add_injectable("calibrated_folder", settings["calibrated_folder"])
        if "initial_run" in settings.keys():
            orca.add_injectable("initial_run", eval(settings["initial_run"]))
        if "multi_level_lcms" in settings.keys():
            orca.add_injectable("multi_level_lcms", eval(settings["multi_level_lcms"]))
        if "segmented_lcms" in settings.keys():
            orca.add_injectable("segmented_lcms", eval(settings["segmented_lcms"]))
        if "capacity_boost" in settings.keys():
            orca.add_injectable("capacity_boost", settings["capacity_boost"])
        if "database_control_totals" in settings.keys():
            orca.add_injectable(
                "use_database_control_totals", eval(settings["database_control_totals"])
            )

# -----------------------------------------------------------------------------------------
# DOWNLOADS DATA FOR REGION
# -----------------------------------------------------------------------------------------
all_local = orca.get_injectable("all_local")
if not all_local:
    storage_client = storage.Client("swarm-test-1470707908646")
    bucket = storage_client.get_bucket("national_block_v2")

    if orca.get_injectable("local_simulation") == False:
        uc_region_code = orca.get_injectable("mpo_id")
        blob = bucket.get_blob("base_data/us/uc_database/regions.csv")
        blob.download_to_filename("configs/regions.csv")
        regions = pd.read_csv(
            "configs/regions.csv", dtype={"region_id": object, "uc_id": object}
        )
        regions = regions[~regions["uc_id"].isnull()].set_index("uc_id")
        uc_region_dict = regions.to_dict()["region_id"]
        print("URBANCANVAS REGION CODE", uc_region_code)
        print("BLOCK MODEL V2 REGION CODE", uc_region_dict[uc_region_code])
        orca.add_injectable("region_code", uc_region_dict[uc_region_code])

region_code = orca.get_injectable("region_code")
calibrated_folder = orca.get_injectable("calibrated_folder")
print("importing datasources for region %s" % region_code)

if len(region_code) == 2:
    region_type = "state"
elif len(region_code) == 5:
    region_type = "county_id"
elif len(region_code) == 8:
    region_type = "mpo"
else:
    region_type = "region"
orca.add_injectable("region_type", region_type)
data_name = '%s_%s_model_data.h5' % (region_type, region_code)
if calibrated_folder == 'custom':
    data_name = 'custom_%s' % (data_name)
# data_name = 'model_data_2017.h5'

orca.add_injectable('data_name', data_name)
print(data_name)

# Downloading the household totals, income rates, and move in rates
hhsize_data_name = "data/hsize_ct_%s.csv" % region_code
hhsize_data = pd.read_csv(hhsize_data_name,
                          dtype={"lcm_county_id": object,
                                 "year": int,
                                 "hh_size": object,
                                 "total_number_of_households": int})
hhsize_data = hhsize_data.set_index("year")
orca.add_table("hsize_ct", hhsize_data)

income_rates_data_name = "data/income_rates_%s.csv" % region_code
income_rates_data = pd.read_csv(income_rates_data_name,
                                dtype={"lcm_county_id": object,
                                       "year": int,
                                       "rate": float})
orca.add_table("income_rates", income_rates_data)


rel_map_data_name = "data/relmap_%s.csv" % region_code
rel_map_data = pd.read_csv(rel_map_data_name).set_index("index")
orca.add_table("rel_map", rel_map_data)

observed_births_data_name = "outputs/calibration/%s/births_over_time_obs.csv" % region_code
observed_births_data = pd.read_csv(observed_births_data_name) 
orca.add_table("observed_births_data", observed_births_data)

observed_fatalities_data_name = "outputs/calibration/%s/mortalities_over_time_obs.csv" % region_code
observed_fatalities_data = pd.read_csv(observed_fatalities_data_name)
orca.add_table("observed_fatalities_data", observed_fatalities_data)

observed_marrital_data_name = "outputs/calibration/%s/marrital_status_over_time_obs.csv" % region_code
observed_marrital_data = pd.read_csv(observed_marrital_data_name)
orca.add_table("observed_marrital_data", observed_marrital_data)

observed_entering_workforce_data_name = "outputs/calibration/%s/entering_workforce_obs.csv" % region_code
observed_entering_workforce_data = pd.read_csv(observed_entering_workforce_data_name)
orca.add_table("observed_entering_workforce", observed_entering_workforce_data)

observed_exiting_workforce_data_name = "outputs/calibration/%s/exiting_workforce_obs.csv" % region_code
observed_exiting_workforce_data = pd.read_csv(observed_exiting_workforce_data_name)
orca.add_table("observed_exiting_workforce", observed_exiting_workforce_data)

# observed_enrollment_data_name = "outputs/calibration/%s/enrollment_over_time_obs.csv" % region_code
# observed_enrollment_data = pd.read_csv(observed_enrollment_data_name)
# orca.add_table("observed_enrollment_data", observed_enrollment_data)


if not all_local:
    if not os.path.exists("data/%s" % data_name):
        print("Downloading model_data from ", "model_data/%s" % data_name)
        blob = bucket.get_blob("model_data/%s" % data_name)
        blob.download_to_filename("./data/%s" % data_name)
        print("Download of model_data.h5 file done")
    else:
        print("Not downloading model_data.h5, since already exists")
else:
    if not os.path.exists("data/%s" % data_name):
        raise OSError("No input data found at data/%s" % data_name)

# -----------------------------------------------------------------------------------------
# LOADS ORCA TABLES FROM H5 FILE
# -----------------------------------------------------------------------------------------

hdf_path = "data/%s" % data_name
hdf_tables = [
    "blocks",
    "households",
    "persons",
    "residential_units",
    "jobs",
    "values",
    "hct",
    "ect",
    "nodes",
    "edges",
    "travel_data",
    "job_flows",
    "household_targets_acs",
    "unit_targets",
    "job_targets",
    "price_targets",
    "job_county_targets_BEA",
    "unit_county_targets_PEP",
    "validation_ect",
    "validation_hct",
    "household_validation_acs",
    "unit_validation",
    "job_validation",
    "school_locations",
    "work_locations"
    
]
mlcm_tables = {"school_locations": ["person_id", "school_id"],
    "work_locations": ["person_id", "work_block_id"]}

for t, c in mlcm_tables.items():
    data = pd.DataFrame(columns=c)
    orca.add_table(t, data)


store = pd.HDFStore(hdf_path)
if "/metadata" in store.keys():
    hdf_tables += ["metadata"]
store.close()

for table_name in hdf_tables:
    step_name = "load_" + table_name
    LoadTable(
        table=table_name,
        source_type="hdf",
        path=hdf_path,
        extra_settings={"key": table_name},
        name=step_name,
    ).run()


if "metadata" not in orca.list_tables():
    metadata = pd.DataFrame(columns=["value"], index=["max_p_id", "max_hh_id"])
    print(orca.list_tables())
    max_p_id = orca.get_table("persons").local.index.max()
    metadata.loc["max_p_id", "value"] = max_p_id
    max_hh_id = orca.get_table("households").local.index.max()
    metadata.loc["max_hh_id", "value"] = max_hh_id
    orca.add_table("metadata", metadata)
    store = pd.HDFStore(hdf_path)
    store["metadata"] = metadata
    store.close()

persons = orca.get_table("persons").local
# breakpoint()
print(persons.columns)
persons["work_block_id"] = "-1"
persons["workplace_taz"] = "-1"
persons["school_id"] = "-1"
persons["school_block_id"] = "-1"
persons["school_taz"] = "-1"

orca.add_table("persons", persons)

# breakpoint()

# Getting income distribution

persons = orca.get_table("persons").local
# Define the intervals for age and education
age_intervals = [0, 20, 30, 40, 50, 65, 900]
education_intervals = [0, 18, 22, 200]

# Define the labels for age and education groups
age_labels = ['lte20', '21-29', '30-39', '40-49', '50-64', 'gte65']
education_labels = ['lte17', '18-21', 'gte22']
# Create age and education groups with labels
persons['age_group'] = pd.cut(persons['age'], bins=age_intervals, labels=age_labels, include_lowest=True).astype(str)
persons['education_group'] = pd.cut(persons['edu'], bins=education_intervals, labels=education_labels, include_lowest=True).astype(str)

# Group by age and education groups and calculate mean and std deviation of earning
income_dist = persons[persons["worker"]==1].groupby(['age_group', 'education_group']).agg(
    data_mean = ('earning', 'mean'),
    data_std = ('earning', 'std')).reset_index()

# Convert to the parameters of the underlying normal distribution
income_dist["mu"] = np.log(income_dist["data_mean"]**2 / np.sqrt(income_dist["data_std"]**2 + income_dist["data_mean"]**2))
income_dist["sigma"] = np.sqrt(np.log(1 + income_dist["data_std"]**2 / income_dist["data_mean"]**2))

orca.add_table("income_dist", income_dist)

# -----------------------------------------------------------------------------------------
# DOWNLOADS CUSTOM SETTINGS IF AVAILABLE
# -----------------------------------------------------------------------------------------

if calibrated_folder == "custom":
    # Custom settings, useful for the definition of time-based accessibility variables
    print("Checking if custom_settings.yaml file exists")
    if not all_local:
        blob = bucket.blob(
            "calibrated_configs/custom/custom_%s_%s/custom_settings.yaml"
            % (region_type, region_code)
        )
        if blob.exists():
            print("Downloading custom_settings.yaml")
            blob.download_to_filename("configs/custom_settings.yaml")
            with open("configs/custom_settings.yaml") as f:
                custom_settings = yaml.load(f, Loader=yaml.FullLoader)
            orca.add_injectable("custom_settings", custom_settings)
    else:
        try:
            with open("configs/custom_settings.yaml") as f:
                custom_settings = yaml.load(f, Loader=yaml.FullLoader)
            orca.add_injectable("custom_settings", custom_settings)
        except OSError:
            raise OSError("No settings found at configs/custom_settings.yaml")

    # Custom output parameters, useful when variables change from default
    print("Checking if custom output_parameters.yaml file exists")
    if not all_local:
        blob = bucket.blob(
            "calibrated_configs/custom/custom_%s_%s/output_parameters.yaml"
            % (region_type, region_code)
        )
        if blob.exists():
            print("Downloading custom output_parameters.yaml")
            blob.download_to_filename("configs/output_parameters.yaml")
    else:
        if not os.path.exists("configs/output_parameters.yaml"):
            raise OSError("No settings found at configs/output_parameters.yaml")

    # Custom calibration settings, useful to refine specifications
    if orca.get_injectable("running_calibration_routine") is True:
        print("Checking if custom pf_vars.yaml file exists")

        if not all_local:
            blob = bucket.blob(
                "calibrated_configs/custom/custom_%s_%s/pf_vars.yaml"
                % (region_type, region_code)
            )
            if blob.exists():
                print("Downloading custom pf_vars.yaml")
                blob.download_to_filename("pf_vars.yaml")
        else:
            if not os.path.exists("pf_vars.yaml"):
                raise OSError("No settings found at ./pf_vars.yaml")

# -----------------------------------------------------------------------------------------
# ADDS AGGREGATION TABLES
# -----------------------------------------------------------------------------------------


def register_aggregation_table(table_name, table_id):
    """
    Generator function for tables representing aggregate geography.
    """

    @orca.table(table_name, cache=True)
    def func(blocks):
        geog_ids = blocks[table_id].value_counts().index.values
        df = pd.DataFrame(index=geog_ids)
        df.index.name = table_id
        return df

    return func


# Aggregate-geography tables
aggregate_geos = [
    ("pumas", "puma10_id"),
    ("counties", "county_id"),
    ("tracts", "tract_id"),
    ("block_groups", "block_group_id"),
    ("region", "region_id"),
]


if "zone_id" in orca.get_table("blocks").local.columns:
    aggregate_geos += [("zones", "zone_id")]


for geog in aggregate_geos:
    register_aggregation_table(geog[0], geog[1])


# -----------------------------------------------------------------------------------------
# DEFINES YEAR INJECTABLE
# -----------------------------------------------------------------------------------------


@orca.injectable("year")
def year():
    default_year = orca.get_injectable("base_year")
    iter_var = orca.get_injectable("iter_var")
    if iter_var is not None:
        return iter_var
    else:
        return default_year


# -----------------------------------------------------------------------------------------
# DEFINES BUILDING TYPES
# -----------------------------------------------------------------------------------------

btypes_dict = {"sf_own": 1, "sf_rent": 2, "mf_own": 3, "mf_rent": 4}
orca.add_injectable("btypes_dict", btypes_dict)
units = orca.get_table("residential_units").local.copy()

if units["building_type_id"].isin(list(btypes_dict.values())).all():
    # btypes already converted
    pass
else:
    units["building_type"] = units["building_type_id"].copy()
    units["building_type"] = units["building_type"] + "_" + units["tenure"]
    if units["building_type"].isin(list(btypes_dict.keys())).all():
        units["building_type_id"] = units["building_type"].map(btypes_dict)
        units = units.drop(columns=["building_type", "tenure"])
        orca.add_table("residential_units", units)
    else:
        raise ValueError("Units table contains unknown building types!")

# -----------------------------------------------------------------------------------------
# DEFINES DEFAULT TARGET VACANCY FOR REAL ESTATE TRANSITION
# -----------------------------------------------------------------------------------------

vacancy = 1 - (
    len(orca.get_table("households")) * 1.0 / len(orca.get_table("residential_units"))
)
orca.add_injectable("vacancy", vacancy * 1.15)


# -----------------------------------------------------------------------------------------
# COMBINE VALIDATION AND FORECAST CONTROL TOTALS
# -----------------------------------------------------------------------------------------

hct = orca.get_table("validation_hct").local.reset_index()
ect = orca.get_table("validation_ect").local.reset_index()

# if orca.get_injectable('segmented_lcms') == True:
#    hct = hct.groupby(['year', 'hh_type'])['total_number_of_households'].sum().reset_index()
#    ect = ect.groupby(['year', 'agg_sector'])['total_number_of_jobs'].sum().reset_index()
# else:
hct = hct.groupby(["year"])["total_number_of_households"].sum().reset_index()
ect = ect.groupby(["year"])["total_number_of_jobs"].sum().reset_index()
try:
    forecast_hct = (
        orca.get_table("hct")
        .local.groupby(["year"])["total_number_of_households"]
        .sum()
        .reset_index()
    )
    forecast_hct = forecast_hct[forecast_hct["year"] >= hct["year"].max()]
    forecast_hct = forecast_hct.set_index("year").diff().iloc[1:].cumsum()
    base_value = hct[hct["year"] == hct["year"].max()].total_number_of_households.item()
    forecast_hct["total_number_of_households"] += base_value
    hct = hct.append(forecast_hct.reset_index())
except Exception:
    max = hct[hct["year"] == hct["year"].max()]
    min = hct[hct["year"] == hct["year"].min()]
    n_years = max["year"].unique().item() - min["year"].unique().item()
    growth_rate = (
        max["total_number_of_households"].sum()
        / min["total_number_of_households"].sum()
    ) ** (1 / n_years) - 1
    for year in range(hct["year"].max() + 1, 2051):
        hh = round(
            hct[hct["year"] == year - 1]["total_number_of_households"].sum()
            * (1 + growth_rate),
            0,
        )
        df = pd.DataFrame(data={"year": [year], "total_number_of_households": [hh]})
        if "hh_type" in hct.columns:
            df["hh_type"] = -1
        hct = hct.append(df)
orca.add_table("hct", hct.set_index("year"))

try:
    forecast_ect = (
        orca.get_table("ect")
        .local.groupby(["year"])["total_number_of_jobs"]
        .sum()
        .reset_index()
    )
    forecast_ect = forecast_ect[forecast_ect["year"] >= ect["year"].max()]
    forecast_ect = forecast_ect.set_index("year").diff().iloc[1:].cumsum()
    base_value = ect[ect["year"] == ect["year"].max()].total_number_of_jobs.item()
    forecast_ect["total_number_of_jobs"] += base_value
    ect = ect.append(forecast_ect.reset_index())
except Exception:
    max = ect[ect["year"] == ect["year"].max()]
    min = ect[ect["year"] == ect["year"].min()]
    n_years = max["year"].unique().item() - min["year"].unique().item()
    growth_rate = (
        max["total_number_of_jobs"].sum() / min["total_number_of_jobs"].sum()
    ) ** (1 / n_years) - 1
    for year in range(ect["year"].max() + 1, 2051):
        jobs = round(
            ect[ect["year"] == year - 1]["total_number_of_jobs"].sum()
            * (1 + growth_rate),
            0,
        )
        df = pd.DataFrame(data={"year": [year], "total_number_of_jobs": [jobs]})
        if "agg_sector" in ect.columns:
            df["agg_sector"] = -1
        ect = ect.append(df)
orca.add_table("ect", ect.set_index("year"))


# -----------------------------------------------------------------------------------------
# ADD ACTIVITYSIM SKIMS DATA
# -----------------------------------------------------------------------------------------
skims = omx.open_file('data/skims_mpo_{}.omx'.format(region_code),'r')
orca.add_injectable('asim_skims', skims)

# Mode Choice Constants (Consider moving them to .yaml file)
# Here as a place holder for now
orca.add_injectable('cost_per_mile', 18.0) # 18 cents per miles
orca.add_injectable('walkThresh', 2.0) #2 miles
orca.add_injectable('walkSpeed', 3.0) #3 miles per hour
orca.add_injectable('bikeThresh', 6.0) #2 miles
orca.add_injectable('bikeSpeed', 12.00) #3 miles per hour
orca.add_injectable('ivt_cost_multiplier', 0.6)
orca.add_injectable('costShareSr2', 1.75)
orca.add_injectable('costShareSr3', 2.50)
orca.add_injectable('short_i_wait_multiplier', 2.0)
orca.add_injectable('waitThresh', 10.00)
orca.add_injectable('long_i_wait_multiplier', 1.0 )
orca.add_injectable('xwait_multiplier', 2.0)
orca.add_injectable('wacc_multiplier', 2.0)
orca.add_injectable('wegr_multiplier', 2.0)
orca.add_injectable('shortWalk', 0.333)
orca.add_injectable('longWalk', 0.667)
orca.add_injectable('tnc_baseline', 2.20)
orca.add_injectable('tnc_cost_minute', 0.24)
orca.add_injectable('tnc_cost_mile', 1.33)
orca.add_injectable('tnc_min_fare', 7.20)
orca.add_injectable('avg_parking_cost', 2.50)
orca.add_injectable('transit_change', 1)


def add_missing_combinations(df):
    # Get the unique values from each index level
    index_values = [df.index.get_level_values(level).unique() for level in range(df.index.nlevels)]

    # Generate all possible pair combinations
    index_pairs = list(product(*index_values))

    # Reindex the DataFrame with all possible combinations
    new_df = df.reindex(index=index_pairs)

    return new_df

@orca.step('update_travel_data')
def update_travel_data(travel_data):
    t = travel_data.local
    t = add_missing_combinations(t)
    orca.add_table('travel_data', t)
# -----------------------------------------------------------------------------------------
# ADD DEMOS TABLES
# -----------------------------------------------------------------------------------------

demos_tables = [
    "graveyard",
    "pop_over_time",
    "ho_over_time",
    "dead_households",
    "households_mv_out",
    "persons_mv_out",
    "household_mv_in",
    "btable",
    "btable_elig",
    "motable",
    "mohtable",
    "pmovein",
    "hhmovein",
    "edu_over_time",
    "age_over_time",
    "income_over_time",
    "kids_move_table",
    "divorce_table",
    "marriage_table",
    "age_dist_over_time",
    "pop_size_over_time",
    "hh_size_over_time",
    "student_counts",
    "birth",
    "mortalities",
    "pmovein_over_time",
    "hhmovein_over_time",
    "student_population",
    "marrital",
    "exiting_workforce",
    "entering_workforce",
    "school_locations",
    "work_locations"
]

for table in demos_tables:
    orca.add_table(table, pd.DataFrame())


# orca.add_injectable("max_p_id", orca.get_table("persons").local.index.max())
# orca.add_injectable("max_hh_id", orca.get_table("households").local.index.max())

orca.add_injectable("persons_local_cols", orca.get_table("persons").local.columns)
orca.add_injectable("households_local_cols", orca.get_table("households").local.columns)

geoid_to_zone = pd.read_csv("data/geoid_to_zone.csv", dtype={"GEOID": str, "zone_id": str})
geoid_to_zone["GEOID10"] = geoid_to_zone["GEOID"].copy()

blocks_districts = pd.read_csv("data/blocks_school_districts_2010.csv")
blocks_districts["UNIFIED_DISTRICT"] = np.where(blocks_districts["SCHOOL_DIST_TYPE"]=="UNIFIED", 1, 0)
blocks_districts["GEOID10"] = ["0"+str(x) for x in blocks_districts["GEOID10_BLOCK"]]
blocks_districts["GEOID10_SD"] = ["0"+str(x) for x in blocks_districts["GEOID10_SD"]]
blocks_districts["DISTRICT_LEVEL"] = blocks_districts.apply(lambda row: (row['GEOID10_SD'], row['DET_DIST_TYPE']), axis=1)
blocks_districts = blocks_districts.merge(geoid_to_zone, how="left", on=["GEOID10"])
blocks_districts = blocks_districts.rename(columns={"zone_id": "school_taz"})
blocks_districts["school_block_id"] = blocks_districts["GEOID10"].copy()

orca.add_table("blocks_districts", blocks_districts)
orca.add_table("geoid_to_zone", geoid_to_zone)


schools_df = pd.read_csv("data/schools_2010.csv", dtype={"GEOID10": str, "SCHOOL_ID": str})
schools_df['CAP_TOTAL_INC'] = schools_df['CAP_TOTAL'] * 1.2
schools_df['REM_CAP'] = schools_df['CAP_TOTAL_INC']
schools_df["GEOID10"] = ["0"+str(x) for x in schools_df["GEOID10"]]
schools_df["GEOID10_SD"] = ["0"+str(x) for x in schools_df["NCESDist"]]
schools_df["school_id"] = schools_df["SCHOOL_ID"].copy()
schools_df = schools_df[~(schools_df["school_id"]=="0000000")].copy()

orca.add_table("schools", schools_df)

# -----------------------------------------------------------------------------------------
# ADD OUTPUT FOLDER
# -----------------------------------------------------------------------------------------
output_folder = "outputs/simulation/%s/" % region_code
if not os.path.exists(output_folder):
    print("Creating output folder")
    os.mkdir(output_folder)
else:
    print("Output path exists!")
orca.add_injectable("output_folder", output_folder)

# ----------------------------------------------------------------------------------------
# ADD NONURBANSIM TEMPLATE MODELS
# -----------------------------------------------------------------------------------------
def read_yaml(path):
    """A function to read YAML file"""
    with open(path) as f:
        config = list(yaml.safe_load_all(f))[0]

    return config
region_code = orca.get_injectable("region_code")
calibrated_folder = orca.get_injectable("calibrated_folder")
skim_source = orca.get_injectable("skim_source")
calibrated_path = os.path.join(
    'calibrated_configs/', calibrated_folder, region_code)
if os.path.exists(os.path.join('configs', calibrated_path, skim_source)):
    calibrated_path = os.path.join(calibrated_path, skim_source)
configs_folder = 'configs/' + calibrated_path if orca.get_injectable('calibrated') else 'estimated_configs'
marriage_model = read_yaml(configs_folder + "/marriage.yml")
orca.add_injectable("marriage_model", marriage_model)
cohabitation_model = read_yaml(configs_folder + "/cohabitation.yaml")
orca.add_injectable("cohabitation_model", cohabitation_model)