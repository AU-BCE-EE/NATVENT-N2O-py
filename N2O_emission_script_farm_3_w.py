"""
File name: N2O_emission_script_farm_3_w.py
Author: Ida Marie Blok Nielsen
Date: December 9, 2025
Course: RnD project

Description:
    This script is used to calculate N2O emission from danish cattle barns for 
    (farm_3_w, where w means that the measurements were taken in the winter period) by using
    the module "N2O_emission_for_all_farms.py" to process and visualize data.


"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from matplotlib.dates import DateFormatter, DayLocator

from N2O_emission_for_all_farms import add_wind_data, conc_table_bg_T_p, conc_table_mg_m3, plot_single_gas_concentration, \
    apply_wind_background_correction, load_csvs_by_date, get_hourly_td_data, plot_valve_data, process_valve_data

import N2O_emission_for_all_farms
import importlib
importlib.reload(N2O_emission_for_all_farms)

# input parameters
data_folder = Path("../data/Picarro/G2509_3")
dates = ["20250221", "20250222", "20250223"]
valves = [1, 3, 7, 8]
valve_labels = {1: "Background Northwest", 3: "Barn sampling",
                7: "Background Southeast", 8: "Background Northeast"}
farm_name = "farm_3_w"
# processing to get clean data
df = load_csvs_by_date(data_folder, dates)
pvd = process_valve_data(df, valves)
plot_valve_data(pvd, valves, valve_labels, y_columns=[
                "N2O_dry", "CO2_dry"], farm_name=farm_name)
# df_filtered = process_N2O_data(
# data_folder=data_folder, dates=dates, farm_name=farm_name, sample_valves=valves, valve_labels=valve_labels)
# plt.savefig
# input parameters
pvd.to_csv("farm_check.csv")
pvd["st"] = pd.to_datetime(pvd["st"])
wrong_data_start = pd.Timestamp("2025-02-21 15:00:00")
wrong_data_end = pd.Timestamp("2025-02-21 17:15:00")
pvd_filtered = pvd[(pvd["st"] < wrong_data_start)
                   | (pvd["st"] > wrong_data_end)]

wind_file = "../data/DMI/farm_3_DMI.csv"
start_time = "2025-02-21 00:00:00"
stop_time = "2025-02-23 23:00:00"
print(wind_file)

# getting hourly averages and adding wind data

hourly_with_wind = add_wind_data(
    pvd_filtered, wind_file, start_time, stop_time)
print(hourly_with_wind.head())
print(hourly_with_wind.head(20))
# input parameters
wind_info_file = pd.read_csv("farm_info_wind.csv")


# creating two new columns with a bg value selected by rules in wind_info_file
df_n2o_with_bg, df_co2_with_bg = apply_wind_background_correction(hourly_with_wind,
                                                                  wind_info_file,
                                                                  farm_name=farm_name
                                                                  )
print(df_n2o_with_bg.head())
print(df_co2_with_bg.columns)

# input parameters
td_combineddata = load_csvs_by_date(Path("../data/Hobo"), dates)
print(td_combineddata.head())
# Define the devices you want to keep
devices_to_keep = [2, 3]

# Filter the DataFrame
td_combineddata = td_combineddata[td_combineddata["Device"].isin(
    devices_to_keep)]

# Save filtered data to check
# td_combineddata.to_csv("temp_data_check_filtered.csv", index=False)

hourly_td_averages = get_hourly_td_data(td_combineddata, start_time, stop_time)
print(hourly_td_averages.columns)

# getting a table with info (bg, temperature and pressure) to calculate N2O and CO2 concentration in mg/m3
table_with_conc_bg_T_p = conc_table_bg_T_p(
    df_n2o_with_bg, df_co2_with_bg, hourly_with_wind, hourly_td_averages, start_time, stop_time)

print(table_with_conc_bg_T_p.columns)

# calculating N2O and CO2 concentration in mg/m3
table_with_conc_mg_m3 = conc_table_mg_m3(table_with_conc_bg_T_p)
conc_table_mg_m3.head()

# save table under outputs
table_with_conc_mg_m3.to_csv("Check_final_table.csv")
table_with_conc_mg_m3.to_excel(
    f"../output/tables_with_conc/N2O_CO2_conc_{farm_name}.xlsx")

# input parameter
gas_column = "ΔCO2_[mg/m3]"

# figure of gas production during time
fig = plot_single_gas_concentration(table_with_conc_mg_m3, gas_column)
plt.show()
safe_column = gas_column.replace("/", "_")
fig.savefig(f"../figs/plots_with_conc/{safe_column}_plot_{farm_name}.png")

# input parameters in [kg]
CO2_production = 2410
total_weight_animals = 115522
gas_columns = ["ΔCO2_[mg/m3]", "ΔN2O_[mg/m3]"]

# mean value for all 3 days
mean_value = table_with_conc_mg_m3[gas_columns].mean()
mean_value["N2O_emission[kg/day/LU]"] = (((mean_value["ΔN2O_[mg/m3]"] / mean_value["ΔCO2_[mg/m3]"]) *
                                          CO2_production) / total_weight_animals) * 500
print(mean_value.head())
mean_value.to_csv(f"../output/emission_tables/emission_3days_{farm_name}.csv")

# mean value for 3 days seperately
table_with_conc_mg_m3["date_only"] = table_with_conc_mg_m3["hour"].dt.date
mean_by_date = table_with_conc_mg_m3.groupby("date_only")[gas_columns].mean()
mean_by_date["N2O_emission[kg/day/LU]"] = (
    ((mean_by_date["ΔN2O_[mg/m3]"] / mean_by_date["ΔCO2_[mg/m3]"]) * CO2_production) / total_weight_animals) * 500
print(mean_by_date)
mean_by_date.to_csv(
    f"../output/emission_tables/emission_per_day_{farm_name}.csv")
