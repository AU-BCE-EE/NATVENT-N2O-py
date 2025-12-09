"""
File name: N2O_emission_for_all_farms.py
Author: Ida Marie Blok Nielsen
Date: December 9, 2025
Course: RnD project

Description:
    This is a module used to process gas concentration data from Picarro gas analyzer and other data to 
    calculate emission of N2O from naturally ventilated dairy barns based on CO2 as a natural tracer gas 
    and tracer ratio method
    see N2O_emission_script_farm_x(_w).py

"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from matplotlib.dates import DateFormatter, DayLocator


# Set working directory to location of this script


def add_midnight_time(x):
    x = str(x)
    if len(x) == 10:  # only date
        return x + " 00:00:00"
    else:
        return x

# using a list of str to find the right csv files and dates getting path csv files and dates from main script
# returning a combined df with data from the csv files


def load_csvs_by_date(folder: Path, dates: list[str]) -> pd.DataFrame:

    csv_files = os.listdir(folder)
    csv_sub = [file for file in csv_files if any(
        date in file for date in dates)]
    dat1 = [pd.read_csv(folder / file) for file in csv_sub]
    combineddata1 = pd.concat(dat1, ignore_index=True)
    return combineddata1

# filter by sample_valves in script, skip the first 5 rows when valve is changing and get rid of bad data


def process_valve_data(df, sample_valves, skip_rows=5):
    filtered = df[df["MPVPosition"].isin(sample_valves)].copy()

    # Identify valve change points
    filtered['valve_change'] = filtered['MPVPosition'] != filtered['MPVPosition'].shift()
    filtered['group_id'] = filtered['valve_change'].cumsum()
    filtered['row_num'] = filtered.groupby('group_id').cumcount()

    # Skip the first N rows after each change
    filtered = filtered[filtered['row_num'] >= skip_rows].copy()

    # Apply final filters
    filtered = filtered[
        (filtered['ChemDetect'] == 0) &
        (filtered['ALARM_STATUS'] == 0) &
        (filtered['INST_STATUS'] != 7)
    ].copy()

    return filtered

# plot two seperate scatter plots for CO2 and N2O respectively with st on the x axis.
# each sample_valve is also represented


def plot_valve_data(df, sample_valves, valve_labels, y_columns, farm_name, x_column="st", figsize=(14, 6), save_folder="../figs/plots_of_valve_data"):

    df[x_column] = pd.to_datetime(df[x_column])
    for y_col in y_columns:
        plt.figure(figsize=figsize)

        for valve in sample_valves:
            valve_data = df[df["MPVPosition"] == valve]
            label = valve_labels.get(valve, f"Valve {valve}")
            plt.scatter(
                valve_data[x_column], valve_data[y_col], label=label, s=20)

        plt.xlabel("Date")
        plt.ylabel(f"{y_col} [ppm]")
        plt.legend(title="Measurement_line")
        plt.xticks(rotation=45)
        ax = plt.gca()
        ax.xaxis.set_major_locator(DayLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

        plt.tight_layout()
        filename = f"{farm_name}_{y_col}_over_time.png"
        filepath = os.path.join(save_folder, filename)
        plt.savefig(filepath)
        print(f"Saved plot at: {filepath}")

# using dt.floor("h") because "st" is a datetime and it rounds down to every hour
# Then grouping the N2O_dry and CO2_dry columns by their MPVPosition and hour
# Then making a mean for each group


def make_hourly_averages(df, columns_to_avg):

    # Ensure datetime flooring works
    df = df.copy()
    df["hour"] = df["st"].dt.floor("h")

    # Group by and average
    hourly_averages = (
        df
        .groupby(["MPVPosition", "hour"])[columns_to_avg]
        .mean()
        .reset_index()
    )

    return hourly_averages

# filter wind_data and pressure data to get the right time change "st" to "hour" to merge them with hourly_averages


def merge_wind_data(hourly_averages, wd_data, start_time, stop_time):

    wd_data = wd_data.copy()
    # "st" to datetime
    wd_data["st"] = pd.to_datetime(wd_data["st"], errors="coerce")

    # Filter by time range
    wd_filtered = wd_data[(wd_data["st"] >= start_time)
                          & (wd_data["st"] <= stop_time)]

    # Rename 'st' to 'hour' to match hourly_averages
    wd_filtered = wd_filtered.rename(columns={"st": "hour"})

    # Merge on 'hour'
    merged = pd.merge(
        hourly_averages,
        wd_filtered[["hour", "mean_wind_dir", "mean_pressure"]],
        on="hour",
        how="left"
    )

    return merged

# build averages of valve positions when needed
# looks for e.g. pos_7_8 in wind_info splitting them and rename it to pos_7 and pos_8 and look
# for those in the dataframe then if pos_7 and pos_8 is in the dataframe it makes an average of those and
# create a column with the name pos_7_8


def build_averages(df, wind_info):
    for pos in wind_info["position"]:
        if "_" in pos:
            parts = pos.replace("pos_", "").split("_")  # ['7', '8']
            if len(parts) > 1:
                pos_cols = [f"pos_{p}" for p in parts]
                df[pos] = df[pos_cols].mean(axis=1)
    return df

# applying wind direction rules to fill a background column e.g. N2O_dry_bg
# extract the mean_wind_dir column, loops over every rule from farm_rules, makes a bollean series
# where if the mean_wind_dir is true then the value is applied to background column


def apply_wind_background(df, farm_rules, gas):

    wind_dir = df["mean_wind_dir"]

    for _, rule in farm_rules.iterrows():
        start = rule["wind_dir_start"]
        end = rule["wind_dir_end"]
        pos = rule["position"]

        if start < end:
            mask = (wind_dir > start) & (wind_dir <= end)
        else:
            mask = (wind_dir > start) | (wind_dir <= end)

        if pos in df.columns:
            df.loc[mask, f"{gas}_bg"] = df.loc[mask, pos]

    return df

# get average of temperature by hour to match the previous df


def get_hourly_td_data(df, start_time, stop_time):

    df = df.copy()

    # Convert to datetime safely
    df["st"] = pd.to_datetime(df["st"], errors="coerce")

    # Filter by date range
    df = df[(df["st"] >= start_time) & (df["st"] <= stop_time)]

    # Floor timestamps to the hour
    df["hour"] = df["st"].dt.floor("h")

    # Group by hour and compute average Temp
    hourly_avg = df.groupby("hour")["Temp"].mean().reset_index()

    return hourly_avg

# getting wind direction from csv file


def add_wind_data(filtered_df, wind_file, start_time, stop_time, gases=("N2O_dry", "CO2_dry")):
    """
    Computes hourly averages and merges with wind data.

    Parameters:
        filtered_df (pd.DataFrame): Filtered valve data from process valve data().
        wind_file (str or Path): Path to the wind data CSV file.
        start_time (str): Start timestamp for filtering wind data (format: 'YYYY-MM-DD HH:MM:SS').
        stop_time (str): Stop timestamp.
        gases (tuple): Gases to average

    Returns:
        pd.DataFrame: Hourly-averaged and wind-merged DataFrame.
    """

    # Step 1: Hourly averages of gas concentrations
    hourly_averages = make_hourly_averages(filtered_df, list(gases))

    # Step 2: Load wind data
    wind_df = pd.read_csv(Path(wind_file))

    # Step 3: Add midnight time to wind data timestamps
    wind_df["st"] = wind_df["st"].apply(add_midnight_time)

    # Step 4: Merge wind data with hourly averages
    df_with_wind_data = merge_wind_data(
        hourly_averages, wind_df, start_time, stop_time)

    return df_with_wind_data


def apply_wind_background_correction(df_with_wind_data, wind_info_df, farm_name):
    """
    Applies wind background correction using farm-specific wind rules.

    Parameters:
        df_with_wind_data (pd.DataFrame): Hourly data merged with wind info.
        wind_info_df (pd.DataFrame): DataFrame with wind direction rules.
        farm_name (str): Name of the farm, e.g., 'farm_1'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: N2O and CO2 corrected DataFrames.
    """
    # Tag valve position for pivot
    df_with_wind_data['valve_pos'] = "pos_" + \
        df_with_wind_data['MPVPosition'].astype(int).astype(str)

    # Pivot to wide format
    df_n2o = df_with_wind_data.pivot(index=(
        'hour', 'mean_wind_dir'), columns='valve_pos', values='N2O_dry').reset_index()
    df_co2 = df_with_wind_data.pivot(index=(
        'hour', 'mean_wind_dir'), columns='valve_pos', values='CO2_dry').reset_index()

    # Get farm-specific rules
    farm_rules = wind_info_df[wind_info_df['farm'] == farm_name].copy()
    farm_rules['position'] = "pos_" + farm_rules['position'].astype(str)

    # Apply corrections
    df_n2o = build_averages(df_n2o, farm_rules)
    df_n2o_with_bg = apply_wind_background(df_n2o, farm_rules, gas="N2O_dry")

    df_co2 = build_averages(df_co2, farm_rules)
    df_co2_with_bg = apply_wind_background(df_co2, farm_rules, gas="CO2_dry")

    return df_n2o_with_bg, df_co2_with_bg


def conc_table_bg_T_p(df_n2o_with_bg, df_co2_with_bg, hourly_data_with_wind, temp_data, start_time, stop_time):
    """
    Merges N2O and CO2 background based on wind data, adds pressure and temperature.

    Parameters:
        df_n2o (pd.DataFrame): Background-corrected N2O data.
        df_co2 (pd.DataFrame): Background-corrected CO2 data.
        hourly_data_with_wind (pd.DataFrame): Original hourly merged data (to get mean_pressure).
        temp_data (pd.DataFrame): temperature sensor data.
        start_time (str): Start of analysis.
        stop_time (str): End of analysis.

    Returns:
        pd.DataFrame: Final merged DataFrame with bg-corrected gases, pressure, and temperature.
    """

    # Rename valve columns for clarity
    df_n2o_with_bg = df_n2o_with_bg.rename(
        columns=lambda x: f"{x}_N2O" if x.startswith("pos_") else x)
    df_co2_with_bg = df_co2_with_bg.rename(
        columns=lambda x: f"{x}_CO2" if x.startswith("pos_") else x)

    # Merge N2O and CO2
    df_with_bg = pd.merge(
        df_n2o_with_bg,
        df_co2_with_bg,
        on=["hour", "mean_wind_dir"],
        how="left"
    )

    # Merge in pressure (after both gases)

    mean_pressure = (
        hourly_data_with_wind[["hour", "mean_pressure"]]
        # Ensure no multiple pressures per hour
        .drop_duplicates(subset="hour")
    )
    df_with_bg = pd.merge(df_with_bg, mean_pressure, on="hour", how="left")

    # Merge temperature into full dataset
    df_with_all_info = pd.merge(
        df_with_bg,
        temp_data,
        on="hour",
        how="left"
    )
    # Add barn difference (e.g., pos_3 is barn)
    df_with_all_info["ΔN2O_barn"] = df_with_all_info["pos_3_N2O"] - \
        df_with_all_info["N2O_dry_bg"]
    df_with_all_info["ΔCO2_barn"] = df_with_all_info["pos_3_CO2"] - \
        df_with_all_info["CO2_dry_bg"]
    # Used for farm 1 when having a DL sample line (pos_2)
    df_with_all_info["ΔN2O_DL"] = df_with_all_info["pos_2_N2O"] - \
        df_with_all_info["N2O_dry_bg"]
    df_with_all_info["ΔCO2_DL"] = df_with_all_info["pos_2_CO2"] - \
        df_with_all_info["CO2_dry_bg"]

    return df_with_all_info


def conc_table_mg_m3(df_with_all_info):
    """ 
    Calculating concentartions to mg/m3 

    Parameters:
        df_with_all_info

    Returns:
        pd.DataFrame: Final table with info + calculated concentrations
    """
    R = 8.3144626e3  # L·Pa/(mol·K)
    M_N2O = 44.013  # g/mol
    M_CO2 = 44.009  # g/mol for CO2

    df_with_all_info["ΔN2O_[mg/m3]"] = (df_with_all_info["ΔN2O_barn"] * M_N2O *
                                        df_with_all_info["mean_pressure"] * 100) / (R * (df_with_all_info["Temp"] + 273.15))
    df_with_all_info["ΔCO2_[mg/m3]"] = (df_with_all_info["ΔCO2_barn"] * M_CO2 *
                                        df_with_all_info["mean_pressure"] * 100) / (R * (df_with_all_info["Temp"] + 273.15))
    # just for farm 1 with DL sampling line
    df_with_all_info["ΔN2O_[mg/m3]_DL"] = (df_with_all_info["ΔN2O_DL"] * M_N2O *
                                           df_with_all_info["mean_pressure"] * 100) / (R * (df_with_all_info["Temp"] + 273.15))
    df_with_all_info["ΔCO2_[mg/m3]_DL"] = (df_with_all_info["ΔCO2_DL"] * M_CO2 *
                                           df_with_all_info["mean_pressure"] * 100) / (R * (df_with_all_info["Temp"] + 273.15))
    return df_with_all_info


def plot_single_gas_concentration(df, column):
    """
    Creates and returns a plot of a single gas or multiple gas concentration over time with a mean line.

    Parameters:
        df (pd.DataFrame): DataFrame with 'hour' and the gas concentration column.
        column (str) or (list): The column to plot.

    Returns:
        matplotlib.figure.Figure: The figure object for further use (e.g., saving).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot single gas data -used for most farms
    # ax.plot(df["hour"], df[column], label=legend_map, color="blue", alpha=0.7)

    # for plotting a legend
    legend_map = {
        "ΔCO2_[mg/m3]": "Averaged CO₂ barn",
        "ΔCO2_[mg/m3]_DL": "Averaged CO₂ Deep Litter area",
        # "ΔN2O_[mg/m3]": "Averaged N₂O barn",
        # "ΔN2O_[mg/m3]_DL": "Averaged N₂O Deep Litter area",
    }

    # Plot multiple gas data (only for farm_1)
    for col in column:
        label_text = legend_map.get(col, col)
        ax.plot(df["hour"], df[col], label=label_text, alpha=0.7)
    # Labels and formatting
    ax.set_xlabel("Date")
    ax.set_ylabel("ΔCO₂ [mg m$^{-3}$]")
    # ax.set_ylabel("ΔN₂O [mg m$^{-3}$]")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    ax = plt.gca()
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.tight_layout()

    return fig
