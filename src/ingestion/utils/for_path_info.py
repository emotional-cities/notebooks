import os
import re
import numpy as np
import pandas as pd
from unidecode import unidecode
import geopandas as gpd
from shapely.geometry import Point, LineString

def fetch_path_num(data_path):
    """Fetch the session number from the data path.
    Extract the session number from the data path.
    It can be exctracted from original data_path or from the session name.
    """
    if '_' in data_path:
        path = str(data_path)
        path = path.split('\\')
        filename = path[-1]
        match = re.search(r'1(\d{2})', filename)
        if match:
            # The group(1) method returns the matched string
            numbers = match.group(1)
            print(numbers)
    else:        
        # Load session information
        sessions = [
            ('Baixa', 4),
            ('Belem', 1),
            ('Parque', 6),
            ('Gulbenkian', 3),
            ('Lapa', 2),
            ('Graca', 5),
            ('Gulb1', 7),
            ('Casamoeda', 8),
            ('Agudo', 9),
            ('Msoares', 10),
            ('Marvila', 11),
            ('Oriente', 12),
            ('Madre', 13),
            ('Pupilos', 14),
            ('Luz', 15),
            ('Alfarrobeira', 16),
            ('Restauradores', 17),
            ('Restelo', 18),
            ('Estrela', 19),
            ('EstrelaA', 20),
            ('EstrelaB', 21),
            ('Prazeres', 22)            
        ]
        # Get number from sessions
        for session_name, session_number in sessions:
            if session_name.lower() in data_path.lower():
                numbers = session_number
    
    return numbers

def extract_session_name(folder_name):
    """Fetch corret session name from the original folder name
    Extract the first string between underscores in the folder name.
    Normalize special characters like 'ç' to 'c'.
    
    Args:
        folder_name (str): Folder name to extract the session name.
    
    Returns:
        str: Normalized session name or the original folder name if no match is found.
    """
    match = re.search(r'_(.*?)_', folder_name)  # Find first string between underscores
    if match:
        session_name = match.group(1)
        return unidecode(session_name)  # Normalize special characters
    return unidecode(folder_name)  # Fallback to the original folder name

def assign_typologies(session):
    # Import necessary functions
    from utils import fetch_path_num
    import utils.for_setpath as path
    import numpy as np
    from shapely.geometry import Point, LineString
    from math import sqrt
    import pyproj
    from shapely.ops import transform
    from functools import partial

    # Get path information
    path_num_ori = fetch_path_num(session)
    path_num = str(path_num_ori).zfill(2)  # make it a two-digit string

    # Dictionary mapping for shapefile names
    path_files = {
        '01': "01_belem.shp",
        '02': "02_lapa.shp",
        '03': "03_gulbenkian.shp",
        '04': "04_Baixa.shp",
        '05': "05_Graca.shp",
        '06': "06_Pnacoes.shp",
        '07': "07_ANovas_Sa_Bandeira.shp",
        '08': "08_ANovas_CMoeda.shp",
        '09': "09_PFranca_Escolas.shp",
        '10': "10_PFranca_Morais_Soares.shp",
        '11': "11_Marvila_Beato.shp",
        '12': "12_PNacoes_Gare.shp",
        '13': "13_Madredeus.shp",
        '14': "14_Benfica_Pupilos.shp",
        '15': "15_Benfica_Moinhos.shp",
        '16': "16_Benfica_Grandella.shp",
        '17': "17_Restauradores.shp",
        '18': "18_Belem_Estadio.shp",
        '19': "19_Estrela_Jardim.shp",
        '20': "20_Estrela_Assembleia.shp",
        '21': "21_Estrela_Rato.shp",
        '22': "22_Estrela_Prazeres.shp"
    }

    # Get correct shp file
    shp_filename = path_files.get(path_num)
    if not shp_filename:
        raise ValueError(f"Invalid path number: {path_num}")
    
    shp_file = os.path.join(path.sourcedata, 'supp', 'interexperimentalpaths_shp', shp_filename)
    gdf = gpd.read_file(shp_file)
    
    # Print CRS information for debugging
    print(f"Original CRS: {gdf.crs}")
    
    # If data is in WGS84 (EPSG:4326), project to a local UTM zone for Portugal (EPSG:3763 - ETRS89 / Portugal TM06)
    if gdf.crs.to_epsg() == 4326:
        gdf = gdf.to_crs(epsg=3763)
        print("Projected to ETRS89 / Portugal TM06")
    
    # Sort by order if available
    if 'order' in gdf.columns:
        gdf = gdf.sort_values('order').reset_index(drop=True)

    # Get the linestring
    linestring = gdf.geometry.iloc[0]
    coords = list(linestring.coords)
    
    print(f"Number of coordinates: {len(coords)}")
    print(f"First few coordinates: {coords[:3]}")
    
    # Create points and calculate cumulative distances
    points = []
    distances = [0.0]
    
    for i in range(len(coords)):
        x, y, z = coords[i]
        points.append(Point(x, y, z))
        
        if i > 0:
            x1, y1, z1 = coords[i-1]
            x2, y2, z2 = coords[i]
            # Calculate 3D Euclidean distance in meters
            dist = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            distances.append(distances[-1] + dist)
    
    # Print some distance statistics for debugging
    print(f"\nDistance statistics:")
    print(f"Total distance: {distances[-1]:.2f} meters")
    print(f"Average segment length: {np.mean(np.diff(distances)):.2f} meters")
    print(f"Min segment length: {np.min(np.diff(distances)):.2f} meters")
    print(f"Max segment length: {np.max(np.diff(distances)):.2f} meters")
    
    # Create new GeoDataFrame with points and distances
    new_gdf = gpd.GeoDataFrame(
        {'geometry': points, 'cumulative_meters': distances},
        crs=gdf.crs
    )

    # Get excel with typologies
    typologies_file = os.path.join(path.sourcedata, 'supp', 'typologies', 'typologies.xlsx')
    typologies_df = pd.read_excel(typologies_file)
    typedf = typologies_df.loc[typologies_df['pathnum'] == path_num_ori].copy()
    
    if typedf.empty:
        print("No matching rows in typologies for this path. Returning new_gdf.")
        return new_gdf

    # Find typology for each point based on cumulative distance
    def find_typology(dist_value):
        row = typedf.loc[(typedf['lowerbound'] <= dist_value) & 
                        (dist_value < typedf['higherbound'])]
        if row.empty:
            return np.nan
        return row.iloc[0]['typology']

    new_gdf['typology'] = new_gdf['cumulative_meters'].apply(find_typology)

    return new_gdf


def assign_typologies2df(session, input_df, plot=True, upsample_distance=1):  # upsample every 5 meters
    """
    Assigns typologies to points by first projecting them onto a reference path,
    upsampling the points, calculating accurate distances, then assigning typologies.
    
    Args:
        session: Session information for path selection
        input_df: DataFrame with a geometry column containing Points
        plot: Boolean to control whether to create visualization plots
        upsample_distance: Distance in meters between upsampled points
    
    Returns:
        DataFrame with added typology column
    """
    from utils import fetch_path_num
    import utils.for_setpath as path
    import numpy as np
    from shapely.geometry import Point, LineString
    from shapely import wkt
    import pyproj
    from shapely.ops import transform
    from functools import partial
    import geopandas as gpd
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    import contextily as ctx

    # Get path information and load reference path
    path_num_ori = fetch_path_num(session)
    path_num = str(path_num_ori).zfill(2)
    
    # Dictionary mapping for shapefile names
    path_files = {
        '01': "01_belem.shp",
        '02': "02_lapa.shp",
        '03': "03_gulbenkian.shp",
        '04': "04_Baixa.shp",
        '05': "05_Graca.shp",
        '06': "06_Pnacoes.shp",
        '07': "07_ANovas_Sa_Bandeira.shp",
        '08': "08_ANovas_CMoeda.shp",
        '09': "09_PFranca_Escolas.shp",
        '10': "10_PFranca_Morais_Soares.shp",
        '11': "11_Marvila_Beato.shp",
        '12': "12_PNacoes_Gare.shp",
        '13': "13_Madredeus.shp",
        '14': "14_Benfica_Pupilos.shp",
        '15': "15_Benfica_Moinhos.shp",
        '16': "16_Benfica_Grandella.shp",
        '17': "17_Restauradores.shp",
        '18': "18_Belem_Estadio.shp",
        '19': "19_Estrela_Jardim.shp",
        '20': "20_Estrela_Assembleia.shp",
        '21': "21_Estrela_Rato.shp",
        '22': "22_Estrela_Prazeres.shp"
    }
    
    shp_filename = path_files.get(path_num)
    if not shp_filename:
        raise ValueError(f"Invalid path number: {path_num}")
    
    shp_file = os.path.join(path.sourcedata, 'supp', 'interexperimentalpaths_shp', shp_filename)
    path_gdf = gpd.read_file(shp_file)
    
    # Convert string geometries if necessary
    input_df = input_df.copy()
    if isinstance(input_df['geometry'].iloc[0], str):
        input_df['geometry'] = input_df['geometry'].apply(wkt.loads)
    
    # Convert to GeoDataFrame and ensure CRS
    input_gdf = gpd.GeoDataFrame(input_df, geometry='geometry', crs="EPSG:4326")
    if path_gdf.crs.to_epsg() == 4326:
        path_gdf = path_gdf.to_crs(epsg=3763)
    input_gdf = input_gdf.to_crs(epsg=3763)
    
    # Get reference linestring
    linestring = path_gdf.geometry.iloc[0]
    
    # Create upsampled points along the reference path
    path_length = linestring.length
    num_points = int(path_length / upsample_distance)
    distances = np.linspace(0, path_length, num_points)
    upsampled_points = [linestring.interpolate(distance) for distance in distances]
    
    # Project each input point to the reference path
    def project_point(point):
        # Find closest upsampled point
        distances = [point.distance(p) for p in upsampled_points]
        closest_idx = np.argmin(distances)
        
        # Get exact projection on the linestring
        linear_ref = linestring.project(point)
        projected_point = linestring.interpolate(linear_ref)
        
        # Calculate cumulative distance along the path
        return pd.Series({
            'projected_point': projected_point,
            'cumulative_meters': linear_ref
        })
    
    # Project all points and calculate distances
    projection_data = input_gdf.geometry.apply(project_point)
    input_gdf['projected_point'] = projection_data['projected_point']
    input_gdf['cumulative_meters'] = projection_data['cumulative_meters']
    
    # Get typologies
    typologies_file = os.path.join(path.sourcedata, 'supp', 'typologies', 'typologies.xlsx')
    typologies_df = pd.read_excel(typologies_file)
    typedf = typologies_df.loc[typologies_df['pathnum'] == path_num_ori].copy()
    
    if not typedf.empty:
        def find_typology(dist_value):
            row = typedf.loc[(typedf['lowerbound'] <= dist_value) & 
                            (dist_value < typedf['higherbound'])]
            if row.empty:
                return np.nan
            return row.iloc[0]['typology']
        
        input_gdf['typology'] = input_gdf['cumulative_meters'].apply(find_typology)
    else:
        print("No matching rows in typologies for this path.")
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Convert to Web Mercator for plotting
        path_gdf_web = path_gdf.to_crs(epsg=3857)
        input_gdf_web = input_gdf.to_crs(epsg=3857)
        
        # Plot 1: Spatial view
        # Plot reference path
        path_gdf_web.plot(ax=ax1, color='blue', label='Reference Path')
        
        # Plot upsampled points
        upsampled_gdf = gpd.GeoDataFrame(
            geometry=upsampled_points, 
            crs=path_gdf.crs
        ).to_crs(epsg=3857)
        upsampled_gdf.plot(ax=ax1, color='gray', markersize=10, alpha=0.3, label='Upsampled Points')
        
        # Plot original and projected points
        input_gdf_web.plot(ax=ax1, color='red', markersize=50, label='Original Points')
        projected_gdf = gpd.GeoDataFrame(
            geometry=input_gdf['projected_point'], 
            crs=input_gdf.crs
        ).to_crs(epsg=3857)
        projected_gdf.plot(ax=ax1, color='green', markersize=50, label='Projected Points')
        
        # Draw projection lines
        for idx, row in input_gdf_web.iterrows():
            proj_point = projected_gdf.geometry.iloc[idx]
            ax1.plot([row.geometry.x, proj_point.x], 
                    [row.geometry.y, proj_point.y],
                    'k--', alpha=0.5)
        
        ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)
        ax1.set_title('Spatial View with Upsampled Reference Points')
        ax1.legend()
        
        # Plot 2: Distance view
        ax2.plot(distances, [0]*len(distances), 'b-', label='Path')
        ax2.plot(distances, [0]*len(distances), 'b|')
        
        # Plot projected points
        ax2.scatter(input_gdf['cumulative_meters'], [0]*len(input_gdf), 
                   color='green', s=100, zorder=5, label='GPS Points')
        
        # Add typology zones
        if not typedf.empty:
            for _, row in typedf.iterrows():
                ax2.axvspan(row['lowerbound'], row['higherbound'], 
                          alpha=0.2, label=f"Typology: {row['typology']}")
        
        ax2.set_title('Distance Along Path')
        ax2.set_xlabel('Meters')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    return input_gdf

import os
import numpy as np
import pandas as pd
import geopandas as gpd

import shapely.wkt
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

from sklearn.neighbors import KDTree  # for fast nearest-neighbor search

def assign_typologies2df_test(session, input_df, upsample_distance=5, plot=True):
    """
    1) Read the path shapefile for the given session.
    2) Up-sample the line at 'upsample_distance' meter intervals 
       => create array of upsampled points with known 'cumulative_meters'.
    3) Build KD-Tree in projected space from these up-sampled points.
    4) For each geometry in input_df, find nearest upsampled point => assign its 'cumulative_meters'.
    5) Use 'typologies.xlsx' to find which typology matches that 'cumulative_meters'.
    6) Return updated DataFrame with columns 'cumulative_meters' and 'typology'.

    Args:
        session (str): e.g. "Lapa", from which we derive the path shape to load.
        input_df (pd.DataFrame): must have 'geometry' column 
                                 with WKT strings or shapely geometry objects.
        upsample_distance (float): distance step (meters) for up-sampling the path line.
        plot (bool): if True, optionally you can add code to visualize (omitted for brevity).
    
    Returns:
        pd.DataFrame with new columns: 
            'cumulative_meters' (float) 
            'typology' (str or NaN if out-of-range).
    """
    from utils import fetch_path_num
    import utils.for_setpath as path
    
    # --------------------------------------------------------
    # 1) Identify the shapefile for the given path num
    # --------------------------------------------------------
    path_num_ori = fetch_path_num(session)  # e.g. returns int like 2
    path_num_str = str(path_num_ori).zfill(2)

    path_files = {
        '01': "01_belem.shp",
        '02': "02_lapa.shp",
        '03': "03_gulbenkian.shp",
        '04': "04_Baixa.shp",
        '05': "05_Graca.shp",
        '06': "06_Pnacoes.shp",
        '07': "07_ANovas_Sa_Bandeira.shp",
        '08': "08_ANovas_CMoeda.shp",
        '09': "09_PFranca_Escolas.shp",
        '10': "10_PFranca_Morais_Soares.shp",
        '11': "11_Marvila_Beato.shp",
        '12': "12_PNacoes_Gare.shp",
        '13': "13_Madredeus.shp",
        '14': "14_Benfica_Pupilos.shp",
        '15': "15_Benfica_Moinhos.shp",
        '16': "16_Benfica_Grandella.shp",
        '17': "17_Restauradores.shp",
        '18': "18_Belem_Estadio.shp",
        '19': "19_Estrela_Jardim.shp",
        '20': "20_Estrela_Assembleia.shp",
        '21': "21_Estrela_Rato.shp",
        '22': "22_Estrela_Prazeres.shp"
    }

    shp_filename = path_files.get(path_num_str, None)
    if not shp_filename:
        raise ValueError(f"No known .shp for path_num={path_num_str} (session={session})")

    shp_path = os.path.join(path.sourcedata, 'supp', 'interexperimentalpaths_shp', shp_filename)
    if not os.path.isfile(shp_path):
        raise FileNotFoundError(f"Cannot find shapefile: {shp_path}")

    # --------------------------------------------------------
    # 2) Load the line geometry, upsample
    # --------------------------------------------------------
    path_gdf = gpd.read_file(shp_path)
    if path_gdf.empty:
        raise ValueError(f"Shapefile {shp_path} is empty or invalid")

    # We assume only 1 geometry in path_gdf
    linestring = path_gdf.geometry.iloc[0]
    if not isinstance(linestring, LineString):
        raise TypeError("The path shapefile geometry is not a LineString")

    # We'll transform to a projected CRS for length/distance calculations
    # often for Lisbon area, EPSG:3763 is used (PT-TM06).
    # If your shapefile is already in 3763, skip the re-projection.
    desired_crs = "EPSG:3763"
    if path_gdf.crs and path_gdf.crs.to_string() != desired_crs:
        path_gdf = path_gdf.to_crs(desired_crs)
        linestring = path_gdf.geometry.iloc[0]  # updated in new CRS
    else:
        # If there's no CRS or already EPSG:3763, we'll assume it's correct
        pass

    # length of the line
    line_length = linestring.length
    # create distances
    n_steps = int(np.floor(line_length / upsample_distance)) + 1
    dist_array = np.linspace(0, line_length, n_steps)
    
    # upsample points
    upsample_coords = []
    for d in dist_array:
        p = linestring.interpolate(d)
        upsample_coords.append((p.x, p.y))

    # We'll store the cumulative distance in parallel
    upsample_cumdist = dist_array.copy()

    # --------------------------------------------------------
    # 3) Build KDTree from upsample_coords
    # --------------------------------------------------------
    # shape = (n_steps, 2)
    upsample_points = np.array(upsample_coords)
    kdtree = KDTree(upsample_points)

    # --------------------------------------------------------
    # 4) Convert input_df to GeoDataFrame in same projected CRS
    # --------------------------------------------------------
    # Convert WKT geometry if needed
    df = input_df.copy()
    if isinstance(df['geometry'].iloc[0], str):
        df['geometry'] = df['geometry'].apply(shapely.wkt.loads)

    # Make a GeoDataFrame
    gdf_input = gpd.GeoDataFrame(df, geometry='geometry')
    # Try reproject
    if gdf_input.crs is None or gdf_input.crs.to_string() != desired_crs:
        gdf_input = gdf_input.set_crs("EPSG:4326", allow_override=True)
        gdf_input = gdf_input.to_crs(desired_crs)

    # --------------------------------------------------------
    # 5) For each point, find nearest upsampled point => assign cumulative distance
    # --------------------------------------------------------
    xvals = gdf_input.geometry.x.values
    yvals = gdf_input.geometry.y.values
    coords_input = np.column_stack([xvals, yvals])  # shape=(N,2)

    dist, idx = kdtree.query(coords_input, k=1)  # nearest neighbor indices
    # idx is shape=(N,), upsample_cumdist is shape=(n_steps,)
    # so each input row i => upsample_cumdist[idx[i]]

    gdf_input["cumulative_meters"] = upsample_cumdist[idx]

    # --------------------------------------------------------
    # 6) Map cumulative distance => typology from excel
    # --------------------------------------------------------
    from utils.for_setpath import sourcedata
    typologies_file = os.path.join(sourcedata, 'supp', 'typologies', 'typologies.xlsx')
    typedf = pd.read_excel(typologies_file)
    typedf = typedf.loc[typedf['pathnum'] == path_num_ori].copy()
    typedf = typedf.sort_values('lowerbound')  # ensure sorted

    # define function
    def find_typology(dist_value):
        row = typedf.loc[(typedf['lowerbound'] <= dist_value) & (dist_value < typedf['higherbound'])]
        if row.empty:
            return np.nan
        return row.iloc[0]['typology']

    if not typedf.empty:
        gdf_input["typology"] = gdf_input["cumulative_meters"].apply(find_typology)
    else:
        print(f"No typologies found for pathnum={path_num_ori}; typology column set to NaN")
        gdf_input["typology"] = np.nan

    # --------------------------------------------------------
    # 7) Optional plot for debugging
    # --------------------------------------------------------
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,8))
        # plot line
        path_gdf.to_crs(desired_crs).plot(ax=ax, color='red', linewidth=2, label='path line')
        # plot upsampled
        ax.scatter(upsample_points[:,0], upsample_points[:,1], s=5, c='blue', label='upsampled points')
        # plot input
        gdf_input.plot(ax=ax, column='cumulative_meters', cmap='viridis', markersize=30, legend=True)
        ax.set_title(f"assign_typologies2df(session={session}): Debug Plot")
        ax.legend()
        plt.show()

    # Convert back to original CRS if you want
    # gdf_input = gdf_input.to_crs("EPSG:4326")

    return pd.DataFrame(gdf_input)


import os
import numpy as np
import pandas as pd
import geopandas as gpd

import shapely.wkt
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

def assign_typologies2df_order(session, input_df, upsample_distance=5, plot=True):
    """
    1) Load the path shapefile for the given session and upsample the line at
       'upsample_distance' intervals to create a reference grid (ground truth) with cumulative distances.
    2) For each GPS point in input_df, compute the Euclidean distance (in EPSG:3763) to every reference point,
       find the nearest reference point (by taking the square root of the sum of squared differences),
       and assign its cumulative distance to that GPS point.
    3) Optionally, produce a diagnostic two-panel plot:
         - Left: The reference grid (upsampled points) over the path.
         - Right: The real GPS points with dashed lines connecting each GPS point to its corresponding reference point,
           with the dash line color coded by the distance between the points.
    
    Args:
        session (str): e.g. "Lapa", from which the path shapefile is derived.
        input_df (pd.DataFrame): must contain a 'geometry' column with WKT strings or shapely geometry objects.
        upsample_distance (float): spacing (in meters) for upsampling the path.
        plot (bool): if True, produce diagnostic plots.
    
    Returns:
        pd.DataFrame: A GeoDataFrame (EPSG:3763) identical to input_df with a new 'cumulative_meters' column.
    """
    # --- standard imports ---
    from utils import fetch_path_num
    import utils.for_setpath as path
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.geometry import LineString, Point
    import shapely.wkt
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    from matplotlib import cm, colors

    DEBUG = True  # Enable debug prints

    # --------------------------------------------------------
    # 1) Load the shapefile and create the reference grid (upsampled points)
    # --------------------------------------------------------
    print("1/4 Loading path...")
    path_num_ori = fetch_path_num(session)
    path_num_str = str(path_num_ori).zfill(2)

    path_files = {
        '01': "01_belem.shp",
        '02': "02_lapa.shp",
        '03': "03_gulbenkian.shp",
        '04': "04_Baixa.shp",
        '05': "05_Graca.shp",
        '06': "06_Pnacoes.shp",
        '07': "07_ANovas_Sa_Bandeira.shp",
        '08': "08_ANovas_CMoeda.shp",
        '09': "09_PFranca_Escolas.shp",
        '10': "10_PFranca_Morais_Soares.shp",
        '11': "11_Marvila_Beato.shp",
        '12': "12_PNacoes_Gare.shp",
        '13': "13_Madredeus.shp",
        '14': "14_Benfica_Pupilos.shp",
        '15': "15_Benfica_Moinhos.shp",
        '16': "16_Benfica_Grandella.shp",
        '17': "17_Restauradores.shp",
        '18': "18_Belem_Estadio.shp",
        '19': "19_Estrela_Jardim.shp",
        '20': "20_Estrela_Assembleia.shp",
        '21': "21_Estrela_Rato.shp",
        '22': "22_Estrela_Prazeres.shp"
    }

    shp_path = os.path.join(path.sourcedata, 'supp', 'interexperimentalpaths_shp', path_files[path_num_str])
    # Read the shapefile and reproject to EPSG:3763 (a metric projection)
    path_gdf = gpd.read_file(shp_path).to_crs("EPSG:3763")
    linestring = path_gdf.geometry.iloc[0]
    print(f"• Path length: {linestring.length:.2f} m | CRS: {path_gdf.crs}")

    print("2/5 Creating reference grid...")
    line_length = linestring.length
    # Create distances along the line (in meters)
    ref_distances = np.arange(0, line_length, upsample_distance)
    ref_points = [linestring.interpolate(d) for d in ref_distances]
    # Get reference coordinates as a NumPy array (EPSG:3763)
    ref_coords = np.array([[p.x, p.y] for p in ref_points])
    print(f"• Reference grid shape: {ref_coords.shape}")

    # --------------------------------------------------------
    # 2) Process GPS points: load and reproject them to EPSG:3763.
    # --------------------------------------------------------
    print("3/5 Processing GPS points...")
    gdf_orig = gpd.GeoDataFrame(
        input_df.copy(),
        geometry=input_df['geometry'].apply(shapely.wkt.loads),
        crs="EPSG:4326"
    )
    # Reproject to the same coordinate system as the reference grid
    gdf = gdf_orig.to_crs("EPSG:3763")
    print(f"• Number of GPS points: {len(gdf)}")

    # --------------------------------------------------------
    # 3) For each GPS point, compute the Euclidean distance to every reference point.
    # --------------------------------------------------------
    print("4/5 Calculating Euclidean distances over full reference grid...")
    cumulative_meters = []
    indices = []      # To store the index of the closest reference point for each GPS point
    min_dists = []    # To store the corresponding minimal distances
    for idx, gps in enumerate(gdf.geometry):
        gps_x, gps_y = gps.x, gps.y
        # Compute differences with every reference point
        differences = ref_coords - np.array([gps_x, gps_y])
        # Compute Euclidean distances: sqrt(dx^2 + dy^2)
        distances = np.sqrt(np.sum(differences**2, axis=1))
        min_idx = np.argmin(distances)
        indices.append(min_idx)
        min_distance = distances[min_idx]
        min_dists.append(min_distance)
        cum_val = ref_distances[min_idx]
        cumulative_meters.append(cum_val)
        if DEBUG and idx < 5:
            print(f"GPS index {idx}: Point ({gps_x:.2f}, {gps_y:.2f}) -> closest ref index {min_idx} with distance {min_distance:.2f} m, cumulative_meters: {cum_val:.2f}")
    
    if DEBUG:
        print("Indices for all GPS points:", indices)
    
    # Assign the computed cumulative distances (no monotonic enforcement here)
    gdf['cumulative_meters'] = cumulative_meters

    # Create a mapping from cumulative_meters (rounded) to the corresponding reference point.
    ref_mapping = {round(d, 6): pt for d, pt in zip(ref_distances, ref_points)}

    # --------------------------------------------------------
    # 4) Create diagnostic plots with two subplots and connection patches.
    #    The dash lines will be color-coded based on the distance between the GPS point and its reference.
    # --------------------------------------------------------
    if plot:
        print("5/5 Creating diagnostic plots with two subplots...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left subplot: Reference grid (upsampled points) and path
        path_gdf.plot(ax=ax1, color='black', linewidth=2, zorder=1)
        ax1.scatter(ref_coords[:, 0], ref_coords[:, 1],
                    c=ref_distances, cmap='viridis', s=20, alpha=0.7, zorder=2)
        ax1.set_title("Upsampled Points (Reference Grid)")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        
        # Right subplot: Real GPS points
        sc = ax2.scatter(gdf.geometry.x, gdf.geometry.y,
                         c=gdf['cumulative_meters'], cmap='viridis', s=50,
                         edgecolor='black', zorder=3)
        ax2.set_title("Real GPS Points")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        
        # Prepare a colormap to color-code the connection patches based on distance.
        # Normalize over the range of computed minimal distances.
        norm = colors.Normalize(vmin=min(min_dists), vmax=max(min_dists))
        cmap = cm.viridis  # Choose your desired colormap

        # Draw ConnectionPatches linking each GPS point (in ax2) to its corresponding reference point (in ax1)
        for idx, row in gdf.iterrows():
            gps_x, gps_y = row.geometry.x, row.geometry.y
            # Get the corresponding minimal distance for this GPS point
            d = min_dists[idx]
            # Map the distance to a color using the normalization and colormap
            line_color = cmap(norm(d))
            # Get the reference point using the stored index
            ref_pt = ref_points[indices[idx]]
            con = ConnectionPatch(xyA=(ref_pt.x, ref_pt.y), coordsA=ax1.transData,
                                  xyB=(gps_x, gps_y), coordsB=ax2.transData,
                                  linestyle="--", color=line_color, alpha=0.8)
            fig.add_artist(con)
        
        # Optionally, add a colorbar to indicate the distance mapping
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # Only needed for older versions of matplotlib
        fig.colorbar(sm, ax=ax2, label="Distance (m)")

        plt.tight_layout()
        plt.show()

        import matplotlib.animation as animation
        from matplotlib.patches import ConnectionPatch
        from matplotlib import cm, colors
        import matplotlib.pyplot as plt

        def create_debug_movie(gdf, ref_points, indices, min_dists, path_gdf, ref_coords, ref_distances,
                            output_filename='debug_movie.mp4', fps=2):
            """
            Creates an animated debug movie that, frame by frame, highlights a real GPS point and its corresponding
            mapped reference (simulated) point. The dashed connection line is color-coded based on the Euclidean distance
            between the two points.

            The animation is saved locally as an MP4 file.

            Args:
            gdf: GeoDataFrame with GPS points (in EPSG:3763) and a 'cumulative_meters' column.
            ref_points: List of simulated (upsampled) points (shapely Points) along the path.
            indices: List of indices (into ref_points) corresponding to each GPS point.
            min_dists: List of the minimum Euclidean distances (in meters) for each GPS point.
            path_gdf: GeoDataFrame of the loaded path (in EPSG:3763).
            ref_coords: NumPy array of shape (N,2) with the coordinates of the reference points.
            ref_distances: NumPy array of distances corresponding to each reference point.
            output_filename (str): Name of the file to save the animation (default 'debug_movie.mp4').
            fps (int): Frames per second for the animation.

            Returns:
            ani: The Matplotlib animation object.
            """
            # Create the figure with two subplots.
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # --- Left subplot: Static background for the reference grid ---
            path_gdf.plot(ax=ax1, color='black', linewidth=2, zorder=1)
            sc1 = ax1.scatter(ref_coords[:, 0], ref_coords[:, 1],
                            c=ref_distances, cmap='viridis', s=20, alpha=0.7, zorder=2)
            ax1.set_title("Upsampled Points (Reference Grid)")
            ax1.set_xlabel("X (m)")
            ax1.set_ylabel("Y (m)")
            
            # --- Right subplot: Static background for real GPS points ---
            sc2 = ax2.scatter(gdf.geometry.x, gdf.geometry.y,
                            c=gdf['cumulative_meters'], cmap='viridis', s=50,
                            edgecolor='black', zorder=3)
            ax2.set_title("Real GPS Points")
            ax2.set_xlabel("X (m)")
            ax2.set_ylabel("Y (m)")
            
            # Prepare a colormap to color-code connection patches based on the distance.
            norm = colors.Normalize(vmin=min(min_dists), vmax=max(min_dists))
            cmap_used = cm.viridis

            # Create highlight markers (initially empty) for the current frame.
            highlight_gps = ax2.scatter([], [], s=150, facecolors='none', edgecolors='red',
                                        linewidths=2, zorder=10)
            highlight_ref = ax1.scatter([], [], s=150, facecolors='none', edgecolors='red',
                                        linewidths=2, zorder=10)
            
            # The update function for the animation.
            def update(frame):
                # Remove the previous connection patch if it exists.
                if hasattr(update, 'con_patch'):
                    update.con_patch.remove()
                
                # Extract the current GPS point and its corresponding reference point.
                current_gps = gdf.geometry.iloc[frame]
                current_ref = ref_points[indices[frame]]
                current_dist = min_dists[frame]
                # Map the distance to a color.
                line_color = cmap_used(norm(current_dist))
                
                # Update highlight markers using set_offsets.
                highlight_gps.set_offsets([[current_gps.x, current_gps.y]])
                highlight_ref.set_offsets([[current_ref.x, current_ref.y]])
                
                # Create a connection patch linking the current reference point to the current GPS point.
                update.con_patch = ConnectionPatch(
                    xyA=(current_ref.x, current_ref.y), coordsA=ax1.transData,
                    xyB=(current_gps.x, current_gps.y), coordsB=ax2.transData,
                    linestyle="--", color=line_color, linewidth=2)
                fig.add_artist(update.con_patch)
                
                # Update the right subplot title to include frame information and the distance.
                ax2.set_title(f"Real GPS Points\nFrame: {frame+1}/{len(gdf)} | Dist: {current_dist:.2f} m")
                return highlight_gps, highlight_ref, update.con_patch

            # Create the animation.
            ani = animation.FuncAnimation(fig, update, frames=len(gdf), interval=1000/fps, blit=False, repeat=True)
            
            # Save the animation locally as an MP4 file using ffmpeg.
            try:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='Matplotlib'), bitrate=1800)
                ani.save(output_filename, writer=writer, dpi=200)
                print(f"Animation successfully saved as '{output_filename}'")
            except Exception as e:
                print("Error saving animation:", e)
            
            plt.tight_layout()
            plt.show()
            return ani

        # Example usage:
        # (Assuming the variables gdf, ref_points, indices, min_dists, path_gdf, ref_coords, ref_distances
        #  have been computed by your processing code.)
        ani = create_debug_movie(gdf, ref_points, indices, min_dists, path_gdf, ref_coords, ref_distances,
                                output_filename='debug_movie.mp4', fps=2)



    return gdf
