# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              IMPORT LIBRARIES                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import matplotlib.pyplot as plt
import numpy as np
from pythermalcomfort.models import utci, solar_gain
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.validation import explain_validity
from shapely.ops import nearest_points, unary_union
import os
import glob
import math
from datetime import datetime
from typing import List, Tuple
from sklearn.neighbors import BallTree
import os
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                          PROCESSING FUNCTIONS                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def geodata_to_csv(dataset, participant_name, session_name, output):
    """Process geodata and save it to a CSV file.
        This is the most important function for processing geodata of each dataset.
        It takes the dataset, participant name, session name, and output path as input.
        It processes the geodata, corrects the GPS data, and saves the geodata to an Excel file.
        It also plots and saves the GPS data to a PNG file.
    """
    
    # Import functions from utils
    from utils import fetch_path_num 
    import utils.for_setpath as path

    print(f"Processing geodata for participant '{participant_name}', session '{session_name}'...")

    try:
        # Create geodata
        geodata = dataset.to_geoframe()

        # Process geodata
        geodata['time'] = geodata.index.to_pydatetime()       # Convert index to datetime
        geodata         = tidy_geodata(geodata)               # Tidy geodata variables
        geodata         = add_environmental_metrics(geodata)  # Add environmental metrics

        # Prepare oututs for the log directory
        os.makedirs(output, exist_ok=True)
        geodata_file = os.path.join(output, f"sub-{participant_name}_ses-{session_name}_geodata.xlsx")
        gps_file     = os.path.join(output, f"sub-{participant_name}_ses-{session_name}_gps.png")

        # Get path information
        path_num = fetch_path_num(session_name)
        path_num = str(path_num).zfill(2) # make it a two-digit string

        # Correct GPS data and plot it
        try: 
            shpdata    = os.path.join(path.sourcedata, 'supp','interexperimentalpaths_shp')
            # Get shapefile name
            if path_num == '01':
                shp_filename = "01_belem.shp"
            elif path_num == '02':
                shp_filename = "02_lapa.shp"
            elif path_num == '03':
                shp_filename = "03_gulbenkian.shp"
            elif path_num == '04':
                shp_filename = "04_Baixa.shp"
            elif path_num == '05':
                shp_filename = "05_Graca.shp"
            elif path_num == '06':
                shp_filename = "06_Pnacoes.shp"
            elif path_num == '07':
                shp_filename = "07_ANovas_Sa_Bandeira.shp"
            elif path_num == '08':
                shp_filename = "08_ANovas_CMoeda.shp"
            elif path_num == '09':
                shp_filename = "09_PFranca_Escolas.shp"
            elif path_num == '10':
                shp_filename = "10_PFranca_Morais_Soares.shp"
            elif path_num == '11':
                shp_filename = "11_Marvila_Beato.shp"
            elif path_num == '12':
                shp_filename = "12_PNacoes_Gare.shp"
            elif path_num == '13':
                shp_filename = "13_Madredeus.shp"
            elif path_num == '14':
                shp_filename = "14_Benfica_Pupilos.shp"
            elif path_num == '15':
                shp_filename = "15_Benfica_Moinhos.shp"
            elif path_num == '16':
                shp_filename = "16_Benfica_Grandella.shp"
            elif path_num == '17':
                shp_filename = "17_Restauradores.shp"
            elif path_num == '18':
                shp_filename = "18_Belem_Estadio.shp"
            elif path_num == '19':
                shp_filename = "19_Estrela_Jardim.shp"
            elif path_num == '20':
                shp_filename = "20_Estrela_Assembleia.shp"
            elif path_num == '21':
                shp_filename = "21_Estrela_Rato.shp"
            elif path_num == '22':
                shp_filename = "22_Estrela_Prazeres.shp"
            elif path_num == '23':
                shp_filename = "23_MAAT_path.shp"
            # Correct GPS data
            shp_file        = os.path.join(shpdata, shp_filename)
            geodata         = correct_gps_data(geodata, shp_file, output, plot=False)
            print(f"Corrected GPS data for participant '{participant_name}', session '{session_name}'...")
            print('Check plot for the corrected GPS data...')
            # Add typology
            if path_num in ['01', '02', '03', '04', '05', '06', '23']:
                print('Adding typology...')
                geodata = add_typology(geodata, path.sourcedata, int(path_num))
            
        except Exception as e:
            print(f"An unexpected error occurred for participant '{participant_name}', session '{session_name}': {e}")
            print("Could not correct GPS data...")

        # Ensure geodata is a DataFrame and save as Excel
        if not isinstance(geodata, pd.DataFrame):
            geodata = pd.DataFrame(geodata)
        geodata.to_excel(geodata_file, index=True)

    except Exception as e:
        print(f"An unexpected error occurred for participant '{participant_name}', session '{session_name}': {e}")
        print("Could not export geodata...")

def tidy_geodata(df):

    """Tidy variables in geodata.
    """
    
    # Define custom parameters
    humidity              = df['tk_humidity_humidity_value'] / 100         # in fraction
    wind_speed            = np.sqrt(df['atmos_northwind_value']**2 + df['atmos_eastwind_value']**2)  # m/s (at ~2.5 m of elevation)
    temp_atmos            = df['atmos_airtemperature_value']               # in ºC
    temp_tk               = df['tk_airquality_temperature_value'] / 100    # in ºC
    temp_tk_ptc           = df['tk_ptc_airtemp_value'] / 100               # in ºC
    temp_radiant          = df['tk_thermocouple_temperature_value'] / 100  # in ºC
    noise_level           = df['tk_soundpressurelevel_spl_value'] /10      # in dBA

    # Assign custom parameters to the df attribute
    df['humidity']        = humidity
    df['wind_speed']      = wind_speed
    df['temp_atmos']      = temp_atmos
    df['temp_tk']         = temp_tk
    df['temp_tk_ptc']     = temp_tk_ptc
    df['temp_radiant']    = temp_radiant
    df['noise_level']     = noise_level

    # Compute the UTCI
    df['utci']            = utci(tdb=temp_atmos, tr=temp_radiant, v=wind_speed, rh=humidity)

    # Get raw GPS coordinates and integrate them into df
    coords                = df.geometry.get_coordinates(include_z=True)
    coords.rename(columns ={'y': 'latitude', 'x': 'longitude', 'z': 'elevation'}, inplace=True)
    df                    = df.join(coords).drop(columns=['geometry'])

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

    # Clamp solar_azimuth to ensure it's within the valid range (0 to 360 degrees)
    df['solar_azimuth'] = df['solar_azimuth'].apply(lambda x: max(0, min(x, 360)))
    solar_gain_output = []
    delta_mrt_values = []

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

def correct_gps_data(df, shp_path, output_dir, plot=True):
    """
    Robust GPS correction with continuity checks, parameter optimization, and advanced visualization.
    Saves all plots and creates a video showing the mapping process.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with latitude and longitude columns
    shp_path : str
        Path to the reference shapefile
    output_dir : str
        Directory where outputs will be saved
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load and validate input
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    ).to_crs("EPSG:3763")

    # 2. Prepare reference path
    path_gdf = gpd.read_file(shp_path).to_crs("EPSG:3763")
    path = path_gdf.geometry.iloc[0]
    
    # Create high-res reference points
    line_length = path.length
    ref_distances = np.arange(0, line_length, 0.1)
    ref_points = np.array([[p.x, p.y] for p in 
                          [path.interpolate(d) for d in ref_distances]])
    tree = BallTree(ref_points)

    def process_points(max_jump, step):
        
        jumps_count = 0
        cumulative_dists = []
        mapped_points = []
        prev_dist = None
        prev_idx = None
        mapping_lines = []
        
        for idx in range(len(gdf)):
            point = gdf.geometry.iloc[idx]
            _, idx_ref = tree.query([[point.x, point.y]], k=1)
            current_idx = idx_ref[0][0]
            current_dist = ref_distances[current_idx]

            if prev_dist is None:
                if current_dist > max_jump*10:
                    jumps_count += 1
                    current_dist = 0
                    current_idx = 0
            else:
                if abs((current_dist - prev_dist)) > max_jump:
                    jumps_count += 1
                    try:
                        current_idx = prev_idx + step
                        current_dist = ref_distances[current_idx]
                    except:
                        current_idx = len(ref_points) - 1
                        current_dist = ref_distances[-1]
            
            mapped_point = ref_points[current_idx]
            mapped_points.append(mapped_point)
            mapping_lines.append([(point.x, point.y), (mapped_point[0], mapped_point[1])])
            
            cumulative_dists.append(current_dist)
            prev_dist = current_dist
            prev_idx = current_idx
        
        return cumulative_dists, jumps_count, np.array(mapped_points), mapping_lines

    # Optimize parameters
    print("Optimizing parameters...")
    steps = range(1, 11)
    max_jumps = np.linspace(1, 100, 100)
    
    min_jumps = float('inf')
    optimal_dists = None
    optimal_params = None
    optimal_mapped_points = None
    optimal_mapping_lines = None
    
    for step in steps:
        for max_jump in max_jumps:
            dists, jumps, mapped_points, mapping_lines = process_points(max_jump, step)
            if jumps < min_jumps:
                min_jumps = jumps
                optimal_dists = dists
                optimal_params = (step, max_jump)
                optimal_mapped_points = mapped_points
                optimal_mapping_lines = mapping_lines
    
    print(f"\nOptimal parameters found:")
    print(f"step: {optimal_params[0]}")
    print(f"max_jump: {optimal_params[1]:.2f}")
    print(f"Number of jumps: {min_jumps}")
    
    # Add corrected coordinates and distances
    gdf['cum_dist'] = optimal_dists
    gdf['corrected_x'] = optimal_mapped_points[:, 0]
    gdf['corrected_y'] = optimal_mapped_points[:, 1]
    gdf['geometry_corrected'] = gpd.points_from_xy(
        gdf.corrected_x, 
        gdf.corrected_y, 
        crs=gdf.crs
    )

    # Transform the corrected geometry back to WGS84
    gdf_wgs84 = gdf.set_geometry('geometry_corrected').to_crs("EPSG:4326")
    
    # Extract the corrected lat/lon coordinates
    gdf['longitude_corrected'] = gdf_wgs84.geometry.x
    gdf['latitude_corrected']  = gdf_wgs84.geometry.y

    if plot:
    
        # Create static plots
        def save_plot(fig, filename):
            fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # 1. Raw vs Corrected Points
        fig, ax = plt.subplots(figsize=(12, 8))
        path_gdf.plot(ax=ax, color='grey', alpha=0.5, label='Reference Path')
        gdf.plot(ax=ax, color='red', alpha=0.5, label='Raw GPS')
        ax.scatter(optimal_mapped_points[:, 0], optimal_mapped_points[:, 1], 
                color='blue', alpha=0.5, label='Corrected Points')
        ax.set_title('Raw vs Corrected GPS Points')
        ax.legend()
        save_plot(fig, 'raw_vs_corrected.png')
        
        # 2. Only Corrected Points
        fig, ax = plt.subplots(figsize=(12, 8))
        path_gdf.plot(ax=ax, color='grey', alpha=0.5, label='Reference Path')
        ax.scatter(optimal_mapped_points[:, 0], optimal_mapped_points[:, 1], 
                color='blue', alpha=0.5, label='Corrected Points')
        ax.set_title('Corrected GPS Points')
        ax.legend()
        save_plot(fig, 'corrected_only.png')
        
        # 3. Cumulative Distance Plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(optimal_dists, '-o', alpha=0.5)
        ax.set_title('Cumulative Distance Along Path')
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Distance (m)')
        ax.grid(True)
        save_plot(fig, 'cumulative_distance.png')
        
        # Create animation
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def init():
            path_gdf.plot(ax=ax, color='grey', alpha=0.5, label='Reference Path')
            gdf.plot(ax=ax, color='red', alpha=0.5, label='Raw GPS')
            ax.legend()
            return []
        
        def update(frame):
            ax.clear()
            path_gdf.plot(ax=ax, color='grey', alpha=0.5, label='Reference Path')
            gdf.iloc[:frame+1].plot(ax=ax, color='red', alpha=0.5, label='Raw GPS')
            
            # Plot corrected points up to current frame
            ax.scatter(optimal_mapped_points[:frame+1, 0], 
                    optimal_mapped_points[:frame+1, 1],
                    color='blue', alpha=0.5, label='Corrected Points')
            
            # Plot mapping line for current point
            line = optimal_mapping_lines[frame]
            ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 
                    'k-', alpha=0.5, label='Mapping' if frame == 0 else "")
            
            ax.set_title(f'Point Mapping Process (Point {frame+1}/{len(gdf)})')
            ax.legend()
            return []
        
        anim = FuncAnimation(fig, update, init_func=init, frames=len(gdf),
                            interval=100, blit=True)
        
        # Save animation
        writer = animation.FFMpegWriter(fps=10, bitrate=1800)
        anim.save(os.path.join(output_dir, 'mapping_process.mp4'), writer=writer)
        plt.close()
    
    return gdf

def group_gps_by_distance(
    gps_data,
    distance_threshold=10
) -> List[Tuple[datetime, datetime]]:
    """
    Groups a GPS+time dataset into segments of ~10 meters. 
    gps_data is assumed to be an iterable of rows with columns:
      ['lat', 'lon', 'time']
    Returns a list of (start_time, end_time) for each ~10 meter segment.
    """
    if len(gps_data) < 2:
        return []

    segments = []
    # Initialize
    seg_start_idx = 0
    cumulative_dist = 0.0

    for i in range(1, len(gps_data)):
        lat1 = gps_data[i-1]['lat']
        lon1 = gps_data[i-1]['lon']
        lat2 = gps_data[i]['lat']
        lon2 = gps_data[i]['lon']

        dist = haversine(lat1, lon1, lat2, lon2)
        cumulative_dist += dist

        if cumulative_dist >= distance_threshold:
            # We define a segment boundary at i
            start_time = gps_data[seg_start_idx]['time']
            end_time   = gps_data[i]['time']
            segments.append((start_time, end_time))

            # Reset boundary
            seg_start_idx = i
            cumulative_dist = 0.0

    # Optionally, if you want to handle the "last partial" segment:
    # if seg_start_idx < len(gps_data)-1:
    #     segments.append((gps_data[seg_start_idx]['time'],
    #                      gps_data[-1]['time']))

    return segments

def add_typology(df, sourcedata, path_num):
    """Add typology for each gps coordinate based on predefined classification of the urban seettings associated with the gps coordinates. The typology information is present in one excel file which contains the intervals in meters associated with each typology. This function associates the GPS coordinates from the path with the typology. It does so by computing the havesine distance for successive GPS coordinates until the distance is greater than the interval in meters associated with the typology. The input should be an excel file with longittude and latitude columns. The output is a new excel file with the typology column added.
    
    Args:
        df (pd.DataFrame): Dataframe with GPS coordinates.
        sourcedata (str): Path to the sourcedata directory.
        path_num (str): Path number.
    
    Returns:
        str: Data path with the typology added.
    """

    # Get excel with typologies
    typologies_file = os.path.join(sourcedata, 'supp', 'typologies', 'typologies.xlsx')
    typologies_df   = pd.read_excel(typologies_file)
    typedf          = typologies_df.loc[typologies_df['pathnum'] == path_num].copy()
    
    if not typedf.empty:
        def find_typology(dist_value):
            row = typedf.loc[(typedf['lowerbound'] <= dist_value) & 
                            (dist_value < typedf['higherbound'])]
            if row.empty:
                return np.nan
            return row.iloc[0]['typology']

        df['typology'] = df['cum_dist'].apply(find_typology)
    else:
        print("No matching rows in typologies for this path.")

    return df


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                   FORMULAS                                    # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def haversine(lat1, lon1, lat2, lon2):
  """
    Calculate the great circle distance between two points
    on the Earth (specified in decimal degrees) using the Haversine formula.
    Source: https://medium.com/@herihermawan/comparing-the-haversine-and-vincenty-algorithms-for-calculating-great-circle-distance-5a2165857666
  """

  # Convert latitude and longitude to radians
  lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

  # Calculate the difference between the two coordinates
  dlat = lat2 - lat1
  dlon = lon2 - lon1

  # Apply the haversine formula
  a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
  c = 2 * math.asin(math.sqrt(a))

  # Calculate the radius of the Earth
  r = 6371 # radius of Earth in kilometers

  # Return the distance
  return c * r

def vincenty(lat1, lon1, lat2, lon2):
  
  """
    Calculate the great circle distance between two points
    on the Earth using the Vincenty formula.
    Source: https://medium.com/@herihermawan/comparing-the-haversine-and-vincenty-algorithms-for-calculating-great-circle-distance-5a2165857666
  """
  # Convert latitude and longitude to radians
  lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

  # Calculate the difference between the two coordinates
  dlat = lat2 - lat1
  dlon = lon2 - lon1

  # Apply the Vincenty formula
  a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

  # Calculate the ellipsoid parameters
  f = 1/298.257223563 # flattening of the Earth's ellipsoid
  b = (1 - f) * 6371 # semi-minor axis of the Earth's ellipsoid

  # Return the distance
  return c * b

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