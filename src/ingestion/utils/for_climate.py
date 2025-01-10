# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              IMPORT LIBRARIES                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import matplotlib.pyplot as plt
import numpy as np
from pythermalcomfort.models import utci, solar_gain
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
import os
import glob

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                          PROCESSING FUNCTIONS                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def tidy_geodata(df):

    """Tidy variables in geodata."""
    
    # Define custom parameters
    humidity = df['tk_humidity_humidity_value'] / 100  # in fraction
    wind_speed = np.sqrt(df['atmos_northwind_value']**2 + df['atmos_eastwind_value']**2)  # m/s (~2.5 m of elevation)
    temp_atmos = df['atmos_airtemperature_value']  # in ºC
    temp_tk = df['tk_airquality_temperature_value'] / 100  # in ºC
    temp_tk_ptc = df['tk_ptc_airtemp_value'] / 100  # in ºC
    temp_radiant = df['tk_thermocouple_temperature_value'] / 100  # in ºC

    # Assign custom parameters to the df attribute
    df['humidity'] = humidity
    df['wind_speed'] = wind_speed
    df['temp_atmos'] = temp_atmos
    df['temp_tk'] = temp_tk
    df['temp_tk_ptc'] = temp_tk_ptc
    df['temp_radiant'] = temp_radiant

    # Compute the UTCI
    df['utci'] = utci(tdb=temp_atmos, tr=temp_radiant, v=wind_speed, rh=humidity)

    # Get GPS coordinates and integrate them into df
    coords = df.geometry.get_coordinates(include_z=True)
    # Optionally rename the coordinate columns
    coords.rename(columns={'y': 'latitude', 'x': 'longitude', 'z': 'elevation'}, inplace=True)
    df = df.join(coords).drop(columns=['geometry'])

    return df


def add_environmental_metrics(df):
    """
    Processes the given dataframe by computing various environmental metrics and appending them to the dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the necessary columns for calculations.

    Returns:
    --------
    pandas.DataFrame
        The original dataframe with new columns added for the calculated metrics.

    The function performs the following computations:
    - Extracts day, month, hour, minute, second from the 'time' column.
    - Calculates the day of the year.
    - Converts mA to W/m^2 (GHI) and sets negative values to 0.
    - Calculates wind speed from north and east wind components.
    - Converts air pressure from Pa to hPa.
    - Calculates dew point temperature.
    - Calculates solar declination.
    - Calculates solar hour angle.
    - Calculates solar altitude and azimuth.
    - Calculates delta_mrt using the pythermalcomfort library.
    - Calculates mean radiant temperature (mrt).
    - Calculates UTCI and the associated stress category.
    """

    # # Ensure 'time' column exists
    # if 'time' not in df.columns:
    #     raise ValueError("'time' column not found in the dataframe.")

    # # Convert the 'time' column to datetime
    # df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # if df['time'].isnull().any():
    #     raise ValueError("Some entries in 'time' column could not be parsed. Please check the format.")

    # Extract date and time components
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['second'] = df['time'].dt.second

    # Calculate day of the year
    df['day_of_year'] = df['time'].dt.dayofyear

    # Define required columns
    required_columns = [
        'tk_dual0_20ma_solarlight_value', 'temp_atmos', 'atmos_northwind_value',
        'atmos_eastwind_value', 'humidity', 'tk_thermocouple_temperature_value',
        'tk_airquality_airpressure_value', 'latitude', 'longitude'
    ]

    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing from the dataframe: {missing_columns}")

    # Calculate GHI (Global Horizontal Irradiance)
    df['ghi'] = 125 * (df['tk_dual0_20ma_solarlight_value'] / 1_000_000 - 4)
    df['ghi'] = df['ghi'].clip(lower=0)  # Set negative GHI values to 0

    # Calculate wind speed
    df['wind_speed'] = np.sqrt(df['atmos_northwind_value']**2 + df['atmos_eastwind_value']**2)

    # Convert air pressure to hPa
    df['hPa'] = df['tk_airquality_airpressure_value'] / 100

    # Calculate dew point temperature
    df['dew_point'] = df['temp_atmos'] - (100 - df['humidity']) / 5

    # Function to calculate solar declination
    def calculate_declination(day_of_year):
        return 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))

    # Function to calculate solar hour angle
    def calculate_hour_angle(hour, minute, second, longitude):
        time_in_hours = hour + minute / 60 + second / 3600
        solar_time = time_in_hours + (longitude / 15)
        return 15 * (solar_time - 12)  # Convert time to degrees

    # Function to calculate solar altitude
    def calculate_solar_altitude(latitude, declination, hour_angle):
        latitude_rad = np.radians(latitude)
        declination_rad = np.radians(declination)
        hour_angle_rad = np.radians(hour_angle)
        altitude_rad = np.arcsin(
            np.sin(latitude_rad) * np.sin(declination_rad) +
            np.cos(latitude_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad)
        )
        return np.degrees(altitude_rad)

    # Function to calculate solar azimuth
    def calculate_solar_azimuth(latitude, declination, hour_angle, solar_altitude):
        latitude_rad = np.radians(latitude)
        declination_rad = np.radians(declination)
        hour_angle_rad = np.radians(hour_angle)
        altitude_rad = np.radians(solar_altitude)

        sin_azimuth = (np.cos(declination_rad) * np.sin(hour_angle_rad)) / np.cos(altitude_rad)
        cos_azimuth = (np.sin(altitude_rad) * np.sin(latitude_rad) - np.sin(declination_rad)) / (
            np.cos(altitude_rad) * np.cos(latitude_rad)
        )

        azimuth_rad = np.arctan2(sin_azimuth, cos_azimuth)
        azimuth_deg = np.degrees(azimuth_rad)
        azimuth_deg = (azimuth_deg + 360) % 360  # Normalize to 0-360 degrees

        # Restrict azimuth to 0-180 degrees
        if azimuth_deg > 180:
            azimuth_deg = 360 - azimuth_deg

        return azimuth_deg

    # Calculate solar declination
    df['declination'] = df['day_of_year'].apply(calculate_declination)

    # Calculate hour angle
    df['hour_angle'] = df.apply(
        lambda row: calculate_hour_angle(row['hour'], row['minute'], row['second'], row['longitude']), axis=1
    )

    # Calculate solar altitude
    df['solar_altitude'] = df.apply(
        lambda row: calculate_solar_altitude(row['latitude'], row['declination'], row['hour_angle']), axis=1
    )

    # Calculate solar azimuth
    df['solar_azimuth'] = df.apply(
        lambda row: calculate_solar_azimuth(
            row['latitude'], row['declination'], row['hour_angle'], row['solar_altitude']
        ), axis=1
    )

    # Calculate delta_mrt using pythermalcomfort's solar_gain function
    delta_mrt_values = []
    for alt, az, ghi in zip(df['solar_altitude'], df['solar_azimuth'], df['ghi']):
        # Ensure valid solar altitude values
        if alt <= 0:
            delta_mrt = 0
        else:
            solar_gain_output = solar_gain(
                sol_altitude=alt,
                sharp=az,
                sol_radiation_dir=ghi,
                sol_transmittance=0.5,
                f_svv=0.5,
                f_bes=0.5,
                asw=0.7,
                floor_reflectance=0.6,
                posture="standing"
            )
            delta_mrt = solar_gain_output['delta_mrt']
        delta_mrt_values.append(delta_mrt)

    # Add delta_mrt to dataframe
    df['delta_mrt'] = delta_mrt_values

    # Calculate mean radiant temperature (mrt)
    df['mrt'] = df['temp_atmos'] + df['delta_mrt']

    # Calculate UTCI and stress category
    utci_values = []
    stress_categories = []
    for tdb, tr, v, rh in zip(df['temp_atmos'], df['mrt'], df['wind_speed'], df['humidity']):
        try:
            utci_output = utci(
                tdb=tdb,
                tr=tr,
                v=v,
                rh=rh,
                units='SI',
                return_stress_category=True
            )
            utci_value = utci_output['utci']
            stress_category = utci_output['stress_category']
        except Exception as e:
            utci_value = np.nan
            stress_category = np.nan
        utci_values.append(utci_value)
        stress_categories.append(stress_category)

    # Add UTCI and stress category to dataframe
    df['utci'] = utci_values
    df['stress_category'] = stress_categories

    return df


def correct_gps_data(df, shp_path):
    """
    Correct GPS coordinates by snapping points to the nearest point on a path.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing 'longitude' and 'latitude' columns
    shp_path (str): Path to the shapefile containing the reference path
    
    Returns:
    geopandas.GeoDataFrame: GeoDataFrame with corrected geometry
    """
    # Create a copy of the input DataFrame to avoid modifications
    df = df.copy()
    
    print("Creating geometry column...")
    # Create a geometry column from 'longitude' and 'latitude'
    df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    
    print("Converting to GeoDataFrame...")
    # Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    
    print("Setting CRS...")
    # Set the Coordinate Reference System (CRS)
    gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)
    
    print(f"Loading shapefile from {shp_path}...")
    # Load the path shapefile
    path_gdf = gpd.read_file(shp_path)
    
    print("Extracting path geometry...")
    # Get the first line from the path shapefile
    path = path_gdf.geometry.iloc[0]
    
    def correct_location(row, path):
        point = row['geometry']
        nearest_point = nearest_points(point, path)[1]
        return nearest_point
    
    print("Correcting points...")
    # Apply the correction to all points
    gdf['geometry'] = gdf.apply(lambda row: correct_location(row, path), axis=1)
    
    print("Returning GeoDataFrame...")
    return gdf  # Make sure this return statement is actually being reached


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                          PLOTTING FUNCTIONS                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def plot_save_gps(gdf, shp_path, output_path):
    """
    Create and save a GPS trajectory plot.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        The GPS data points
    shp_path : str
        Path to the reference shapefile
    output_path : str
        Where to save the plot
    """
    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot reference path
    path_gdf = gpd.read_file(shp_path)
    path_gdf.plot(ax=ax, color='black', linewidth=2, alpha=0.5, label='Reference Path')
    
    # Plot GPS points
    gdf.plot(ax=ax, color='red', markersize=5, alpha=0.6, label='GPS Trajectory')
    
    # Add basic elements
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Save and close
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()