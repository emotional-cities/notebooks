import numpy as np
import pandas as pd
from pythermalcomfort.models import utci, solar_gain

def tidy_geodata(geodf):

    """Tidy variables in geodata."""
    
    # Define custom parameters
    humidity = geodf['tk_humidity_humidity_value'] / 100  # in fraction
    wind_speed = np.sqrt(geodf['atmos_northwind_value']**2 + geodf['atmos_eastwind_value']**2)  # m/s (~2.5 m of elevation)
    temp_atmos = geodf['atmos_airtemperature_value']  # in ºC
    temp_tk = geodf['tk_airquality_temperature_value'] / 100  # in ºC
    temp_tk_ptc = geodf['tk_ptc_airtemp_value'] / 100  # in ºC
    temp_radiant = geodf['tk_thermocouple_temperature_value'] / 100  # in ºC

    # Assign custom parameters to the geodf attribute
    geodf['humidity'] = humidity
    geodf['wind_speed'] = wind_speed
    geodf['temp_atmos'] = temp_atmos
    geodf['temp_tk'] = temp_tk
    geodf['temp_tk_ptc'] = temp_tk_ptc
    geodf['temp_radiant'] = temp_radiant

    # Compute the UTCI
    geodf['utci'] = utci(tdb=temp_atmos, tr=temp_radiant, v=wind_speed, rh=humidity)

    # Get GPS coordinates and integrate them into geodf
    coords = geodf.geometry.get_coordinates(include_z=True)
    # Optionally rename the coordinate columns
    coords.rename(columns={'y': 'latitude', 'x': 'longitude', 'z': 'elevation'}, inplace=True)
    geodf = geodf.join(coords).drop(columns=['geometry'])

    return geodf


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

