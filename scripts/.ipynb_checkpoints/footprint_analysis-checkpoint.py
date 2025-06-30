import pandas as pd
import calc_footprint_FFP_climatology as myfootprint_climato
from pyproj import Transformer
import numpy as np
from windrose import WindroseAxes
import tempfile
from matplotlib import pyplot as plt
import folium
import geopandas as gpd
from shapely.geometry import Polygon
import xarray as xr

def extract_FFP_inputs(df) :
    """
    Extract the columns needed for footprint calculation (adapated to Lautaret ICOS station)

    Inputs: Dataframe containing the required columns

    Ouputs: Dictionary ready for footprint calculation (FFP model, Kljun et al)
    """
    return {
        'zmt':df["zm"],
        'z0t':None,
        'umeant':df["WS_1_1_1"],
        'ht':df["H_RANDOM"],
        'olt':df["MO_LENGTH_1_1_1"],
        'sigmavt':df["V_SIGMA_1_1_1"],
        'ustart':df["USTAR_1_1_1"],
        'wind_dirt':df["WD_2_1_1"]}

def run_FFP_90(df) :
    '''
    Plot the footprint climatology with 10 to 90% contours

    Inputs: Dataframe containing the required columns

    Ouputs: Dictionary with FFP outputs
    '''
    inputs = extract_FFP_inputs(df)
    return myfootprint_climato.FFP_climatology(
        zm=inputs['zmt'].tolist(),
        z0=inputs['z0t'],
        umean=inputs['umeant'].tolist(),
        h=inputs['ht'].tolist(),
        ol=inputs['olt'].tolist(),
        sigmav=inputs['sigmavt'].tolist(),
        ustar=inputs['ustart'].tolist(),
        wind_dir=inputs['wind_dirt'].tolist(),
        domain=[-500., 500., -500., 500.],
        dx=10,
        dy=10,
        rs=[10,20,30,40,50,60,70,80,90],
        verbosity=0,
        fig=1
    )

def run_FFP_nc(df) :
    '''
    Plot the footprint climatology with 10 to 80% contours

    Inputs: Dataframe containing the required columns

    Ouputs: Dictionary with FFP outputs
    '''
    inputs = extract_FFP_inputs(df)
    return myfootprint_climato.FFP_climatology(
        zm=inputs['zmt'].tolist(),
        z0=inputs['z0t'],
        umean=inputs['umeant'].tolist(),
        h=inputs['ht'].tolist(),
        ol=inputs['olt'].tolist(),
        sigmav=inputs['sigmavt'].tolist(),
        ustar=inputs['ustart'].tolist(),
        wind_dir=inputs['wind_dirt'].tolist(),
        domain=[-700., 700., -700., 700.],
        dx=2.5,
        dy=2.5,
        rs= None,
        smooth_data=0,
        verbosity=0,
        fig=0
    )

def transform_coordinates(*args, crs_in, crs_out):
    """
    Transform coordinates between CRS.
    
    Inputs:
        args (float): Coordinates.
        crs_in (str): Input CRS.
        crs_out (str): Output CRS.
    
    Outputs:
        tuple: Transformed coordinates.
    """
    transformer = Transformer.from_crs(crs_in, crs_out, always_xy=True)
    return transformer.transform(*args)
    
def contour_coord_extract(data_FFP, lon_tower, lat_tower) :
    """
    Extracts the coordinates for each r (percentage of source area) and converts it into longitude and latitude

    Inputs:
        data_FFP : Output dictionary of FFP model
        lon_tower, lat_tower : longitude and latitude of the tower

    Outputs: Dictionary with the contour coordinates (in WGS84) for each percentage value
    """
    x0, y0 = transform_coordinates(lon_tower, lat_tower, crs_in = "EPSG:4326", crs_out = "EPSG:32631")
    
    rs_array = np.array(data_FFP['rs'])
    xr_all = data_FFP['xr']
    yr_all = data_FFP['yr']
    
    contours = {}
    for i, r in enumerate(rs_array):
        xr_i = xr_all[i]
        yr_i = yr_all[i]

        # Skip if data is None or invalid
        if xr_i is None or yr_i is None:
            print(f"Skipping {r*100:.0f}% contour (None value).")
            continue

        try:
            xr = np.array(xr_i)
            yr = np.array(yr_i)
            xcoord = x0 + xr
            ycoord = y0 + yr

            lon, lat = transform_coordinates(xcoord, ycoord, crs_in = "EPSG:32631", crs_out = "EPSG:4326")
            contours[r] = (lon, lat)
        except IndexError:
            print(f"Missing {r*100:.0f}% contour.")
    
    return contours

def plot_windrose(data, wind_direction, wind_speed) :
    '''
    Plot the windrose and save it in a temporary file

    Inputs:
        data : dataframe with wind charcateristics (FFP inputs)
        wind_direction, wind_speed : names of the columns related

    Ouputs : Windrose saved in a temporary .png file
    '''
    ax = WindroseAxes.from_ax()
    ax.bar(data[wind_direction], data[wind_speed], normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()

    # Save in a temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp_file.name, bbox_inches='tight')
    plt.close()

    return tmp_file.name

def map_footprint(data, lat_tower, lon_tower, contours_coord) :
    '''
    Create a html map with footprint contour lines and windrose

    Inputs :
        data : dataframe with wind charcateristics (FFP inputs)
        lat_tower, lon_tower : Coordinates (latitude and longitude) of ICOS tower
        contours_coord : Coordinates (latitude and longitude) of footprint contour lines in WGS84

    Ouputs : html map with footprint contour lines and windrose, on a background from GeoPortail
    '''
    windrose_path = plot_windrose(data,'WD_2_1_1', 'WS_1_1_1')
    
    # Get the map from Geoportail and add a marker for the ICOS tower
    map = folium.Map([lat_tower, lon_tower], tiles="GeoportailFrance.plan", zoom_start =17, control_scale = True)
    folium.Marker(
        location = [lat_tower, lon_tower], 
        icon = folium.Icon(icon="cloud"),
    ).add_to(map)

    # Add polygons on the map
    for r, (lon, lat) in contours_coord.items():
    
        points = list(zip(lat, lon))
    
        folium.Polygon(
            locations=points,
            color='blue',
            fill=False
        ).add_to(map)

        # Add the windrose to the map
    folium.Marker(
        location=[45.042969 , 6.405505],
        icon=folium.CustomIcon(
        icon_image=windrose_path,
        icon_size=(200,200), 
        icon_anchor=(25, 25)  # Center the image
        ),
        tooltip="Windrose"
    ).add_to(map)

    return map

def contours_to_gdf(contours_dict, crs="EPSG:4326"):
    '''
    Convert contours coordinates (WGS84) into GeoDataframe (polygons)
    '''
    polygons = []
    labels = []
    for label, (lon, lat) in contours_dict.items():
        poly = Polygon(zip(lon, lat))
        polygons.append(poly)
        labels.append(label)
    gdf = gpd.GeoDataFrame({'label': labels, 'geometry': polygons}, crs=crs)
    return gdf
    
def FFPdict_to_nc(footprint, attrs={}):
    '''
    Convert a footprint climatology dictionary into a netCDF file

    Inputs : footprint : Footprint climatology dictionary

    Ouputs : A netCDF file with global and variable attributes and 2d grid + footprint values
    '''
    fp_value = footprint['fclim_2d']
    x_2d = footprint['x_2d']
    y_2d = footprint['y_2d']
    
    # Extract 1d axis
    x = x_2d[0, :]  
    y = y_2d[:, 0]  

    data = xr.Dataset(
        data_vars={'footprint': (['x', 'y'], fp_value)},
        coords={'x': x, 'y': y}
    )

    data.attrs = {
        'Title': 'Single flux footprint ICOS Lautaret',
        'Summary': "Flux footprint integrated over 30 min from the given Timestamp, for the ICOS associated flux tower of Col du Lautaret, France",
        'Subjects': 'Flux footprint, grassland CO2 flux, ICOS',
        'Creation_Date': '2025-05-23',
        'Conventions': 'CF-1.8',
        'Model_Used': 'FFP, Kljun et al. (2015), doi:10.5194/gmd‐8‐3695‐2015',
        'Creator': 'Alise ROBERT, Institute of Environmental Geosciences (IGE)',
        'Institution': 'Institute of Environmental Geosciences (IGE), Grenoble, France',
        'Contact': 'Alise ROBERT, CHIANTI, IGE, alise.robert@univ-grenoble-alpes.fr',
        'Aknowledgement': 'This work is the result of a M2 internship supervised by Didier VOISIN, didier.voisin@univ-grenoble-alpes.fr',
        'Variables': 'X, Y, Footprint Climatology',
        'Coordinate_Reference_System': "",  # 'WGS 84',
        'Tower_Location_Latitude': 45.041375,
        'Tower_Location_Longitude': 6.410519,
        'Tower_Location_CRS': 'WGS 84',
    }
    data['x'].attrs = {
        'long_name': 'x coordinate of projection',
        'standard_name': 'projection_x_coordinate',
        'units': 'meters'
    }
    data['y'].attrs = {
        'long_name': 'y coordinate of projection',
        'standard_name': 'projection_y_coordinate',
        'units': 'meters'
    }
    data['footprint'].attrs = {
        'long_name': 'single footprint value',
        'units': 'per squared meter'
    }
        
    return data

def data_to_footprint_nc(flux_NDVI_data, attrs={}):
    '''
    Convert a dataframe containing the required columns for footprint calculation into a netCDF file with all the footprint values for each timestamp

    Inputs : flux_NDVI_data : Dataframe with required columns for footprint calculation + NDVI phases + Night info

    Ouputs : A netCDF file (ready to be saved) with footprint values on a x/y grid, for each timestamp and the characteristics of timestamps (night or day-time/vegetation phase)
    '''
    # For every lign of flux_NDVI dataframe
    footprints = []
    timestamps = []
    nights = []
    phases = []
    x = y = None
    
    for idx, row in flux_NDVI_data.iterrows():
        try:
            flux_data = row.to_frame().T
            # Calculate footprint, with dx = dy = 2.5 m
            ffp = run_FFP_nc(flux_data)
    
            if (ffp is not None and not np.all(ffp['fclim_2d'] == 0)):
                # Convert output dictionary into netCDF file
                footprints.append(ffp['fclim_2d'][np.newaxis, :, :])
                timestamps.append(row['TIMESTAMP_START'])
                nights.append(str(row['Night']))
                phases.append(str(row['phase_label']))
    
                # Extract 1d axis, just one time
                if x is None or y is None:
                    x = ffp['x_2d'][0, :]
                    y = ffp['y_2d'][:, 0]
    
        except Exception as e:
            print(f"Error to lign {idx} : {e}")
    
    
    # Stack footprints along time
    footprint_array = np.concatenate(footprints, axis=0)
    
    # Build a dataset
    data = xr.Dataset(
        data_vars={'footprint': (['timestamp', 'x', 'y'], footprint_array),
                   'night' : (['timestamp'], nights),
                   'phase' : (['timestamp'], phases)
        },
        coords={'x': x, 'y': y, 'timestamp':timestamps}
    )
    
    fixed_timestamps = pd.to_datetime(data['timestamp'].values).values.astype('datetime64[ns]')
    
    # Reassign
    data = data.assign_coords(timestamp=fixed_timestamps)
    
    # Metadata
    data.attrs = {
        'Title': 'Single flux footprints ICOS Lautaret (2019-2023)',
        'Summary': "Flux footprints integrated over 30 min from 2019 to 2023, for the ICOS associated flux tower of Col du Lautaret, France",
        'Subjects': 'Flux footprint, grassland CO2 flux, ICOS',
        'Creation_Date': '2025-06-24',
        'Conventions': 'CF-1.8',
        'Model_Used': 'FFP, Kljun et al. (2015), doi:10.5194/gmd‐8‐3695‐2015',
        'Creator': 'Alise ROBERT, Institute of Environmental Geosciences (IGE)',
        'Institution': 'Institute of Environmental Geosciences (IGE), Grenoble, France',
        'Contact': 'Alise ROBERT, CHIANTI, IGE, alise.robert@univ-grenoble-alpes.fr',
        'Aknowledgement': 'This work is the result of a M2 internship supervised by Didier VOISIN, didier.voisin@univ-grenoble-alpes.fr',
        'Variables': 'X, Y, Timestamp, Night, Phase Label, Footprint Climatology',
        'Coordinate_Reference_System': "",  # 'WGS 84',
        'Tower_Location_Latitude': 45.041375,
        'Tower_Location_Longitude': 6.410519,
        'Tower_Location_CRS': 'WGS 84',
    }
    
    data['x'].attrs = {
        'long_name': 'x coordinate of projection',
        'standard_name': 'projection_x_coordinate',
        'units': 'meters'
    }
    data['y'].attrs = {
        'long_name': 'y coordinate of projection',
        'standard_name': 'projection_y_coordinate',
        'units': 'meters'
    }
    data['footprint'].attrs = {
        'long_name': 'single footprint value',
        'units': 'per squared meter'
    }
    
    data['timestamp'].attrs = {
        'long_name': 'timestamp start for the 30 min integration period',
        'standard_name': 'timestamp',
        'unit': 'ns',
        'ancillary' : 'UTC+0'
    }

    data['night'].attrs = {
        'long_name': 'timestamp start for the 30 min integration period',
        'units': 'True for night-time, False for day-time',
        'ancillary' : 'use of astral package'
    }
    
    data['phase'].attrs = {
        'long_name': 'vegetation phase based on NDVI analysis',
        'units': 'True for night-time, False for day-time',
    }
    
    return data

def data_to_daily_nc(flux_NDVI_data, output_path):
    '''
    Convert a dataframe containing the required columns for footprint calculation into netCDF files for each day

    Inputs : 
        flux_NDVI_data : Dataframe with required columns for footprint calculation + NDVI phases + Night info
        output_path : path to store the netCDF files

    Ouputs : Save a netCDF file for each day with footprint values on a x/y grid, for each timestamp and the characteristics of timestamps (night or day-time/vegetation phase)
    '''
    flux_NDVI_data['TIMESTAMP_DATE'] = pd.to_datetime(flux_NDVI_data['TIMESTAMP_START']).values.astype('datetime64[D]')
    unique_dates = flux_NDVI_data['TIMESTAMP_DATE'].drop_duplicates()
    
    for date in unique_dates:
        day_data = flux_NDVI_data[flux_NDVI_data['TIMESTAMP_DATE'] == date]
    
        try:
            footprints_nc = data_to_footprint_nc(day_data)
            myencoding = {
                'x': {'dtype': 'float64', '_FillValue': None},
                'y': {'dtype': 'float64', '_FillValue': None},
                'footprint': {'dtype': 'float64'},
                'night': {'dtype': str},
                'phase': {'dtype': str}
            }
        
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            # Save into netCDF
            filename = f'{output_path}/footprint_{date_str}.nc'
            footprints_nc.to_netcdf(filename, encoding=myencoding)
        except Exception as e:
            print(f'Error for date {date}: {e}')
    
def get_contour(footprint, dx, dy, rs, 
                _get_contour_levels=myfootprint_climato.get_contour_levels,
                _get_contour_vertices=myfootprint_climato.get_contour_vertices):
    '''
    Function to get contour line coordinates 

    Inputs :
        footprint : Structure array (dictionary) with footprint climatology
        dx, dy : Cell size of domain (m)
        rs : Percentage of source area for which to provide contours

    Outputs :
        Footprint dictionary updated with contour lines
        
    '''
    clevs = _get_contour_levels(footprint["fclim_2d"], dx, dy, rs)
    frs = [item[2] for item in clevs]
    xrs = []
    yrs = []
    for ix, fr in enumerate(frs):
        xr, yr = _get_contour_vertices(
            footprint["x_2d"], footprint["y_2d"], footprint["fclim_2d"], fr)
        if xr is None:
            frs[ix] = None
        xrs.append(xr)
        yrs.append(yr)

    footprint.update({"xr": xrs, "yr": yrs, 'fr': frs, 'rs': rs})
    return footprint
