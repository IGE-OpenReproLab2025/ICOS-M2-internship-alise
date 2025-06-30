import glob
import re
import os
import pandas as pd
import numpy as np
from astral import LocationInfo
from astral.sun import sun
from pytz import timezone

def read_csv_file(path, *args, **kwargs):
    """
    Read data from a file

    Inputs:
        path (str with glob characters) : Path to the files
        *args: Additional arguments to pass to the read function.
        **kwargs: Additional keyword arguments to pass to the read function.
            
    Outputs:
        Data read from the file.
    """
    files = glob.glob(path)
    
    # Function to extract the first 4-digit number (year)
    def extract_year(filename):
        match = re.search(r'(\d{4})',os.path.basename(filename))
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Year not found in filename: {filename}")

    # Sort files by extracted year
    files_sorted = sorted(files,key=lambda f: extract_year(f))

    # Read and store DataFrames
    liste = [pd.read_csv(f, *args, **kwargs) for f in files_sorted]

    return liste
    
def prep_timestamp(data_all) :
    """
    Standardize the timestamp column

    Inputs: 
        List of dataframes

    Outputs:
        List of dataframes with "TIMESTAMP_START" at date format
    """
    result = []
    for data in data_all :
        data = data.rename(columns={'TIMESTAMP START': 'TIMESTAMP_START'})
        data['TIMESTAMP_START'] = pd.to_datetime(data['TIMESTAMP_START'], format="%Y%m%d%H%M")
        result.append(data)
    return result

def add_lofreqcol(df_flux, df_lofreq, columns):
    """
    Add given columns of Lofreq dataframes to Flux dataframes

    Inputs:
        df_flux : List of flux dataframes
        df_lofreq : List of low freq dataframes
        columns : columns to add

    Outputs:
        List of flux dataframes with the corresponding columns of lofreq dataframes
    """
    result = []
    for flux, lofreq in zip(df_flux, df_lofreq):
        merged = pd.merge(
            flux,
            lofreq[['TIMESTAMP_START'] + columns],
            on='TIMESTAMP_START',
            how='left'
        )
        result.append(merged)
    return result

def is_night(timestamp):
    """
    Returns True if the time input is during the night

    Inputs:
        timestamp : Datetime column

    Output:
        "Night" column with True for nighttime and False for daytime
    """
    site = LocationInfo(name="Lautaret", region="FR", timezone="Europe/London",
                    latitude=45.04128, longitude=6.41058)
    london = timezone("Europe/London")

    # Add time zone if missing
    if timestamp.tzinfo is None:
        timestamp = london.localize(timestamp)
    s = sun(site.observer, date=timestamp.date(), tzinfo=site.timezone)
    return not (s['sunrise'] <= timestamp <= s['sunset'])

def detect_phase(df, ndvi_col='NDVI_smoothed', diff_thresh=0.003, ndvi_thresh=0.15):
    '''
    Calculate the derivative and detect the vegetation different phases (growth, decline, stagnation, snow)

    Inputs:
        df : Dataframe with NDVI values

    Outputs:
        Dataframe with raw phases and smoothed phases ('phase_label')
    '''

    df = df.copy()
    
    df['NDVI_diff'] = df[ndvi_col].diff()
    df['phase_raw'] = 'Snow'

    mask_growth = df['NDVI_diff'] > diff_thresh
    mask_decline = df['NDVI_diff'] < -diff_thresh
    mask_stagnation = (~mask_growth) & (~mask_decline) & (df[ndvi_col] > ndvi_thresh)

    df.loc[mask_growth, 'phase_raw'] = 'Growth'
    df.loc[mask_decline, 'phase_raw'] = 'Decline'
    df.loc[mask_stagnation, 'phase_raw'] = 'Stagnation'

    # Grouping of identical consecutive phases
    phase_group = (df['phase_raw'] != df['phase_raw'].shift()).cumsum()
    phase_label = df.groupby(phase_group)['phase_raw'].transform('first')

    # Remove the micro-phases < N days
    min_duration = 10  # minimal number of days to keep a phase
    valid = df.groupby(phase_group)[ndvi_col].transform('count') >= min_duration
    phase_label.loc[~valid] = np.nan
    phase_label = phase_label.ffill().bfill()

    df['phase_label'] = phase_label

    return df

def dominant_wind(data, wd_col, sectors = 8):
    '''
    Isolate prevailing winds based on their direction (when their frequency is maximum or close to the maximum)

    Inputs:
        data : dataframe with wind directions
        wd_col : name of wind direction column

    Outputs:
        data_dominant : only the dominant wind(s) ligns from the input dataframe
        Print the prevailing winds directions and frequencies
    '''
    data = data.copy()

    # Center the sectors
    adjusted_wd = (data[wd_col] + 22.5) % 360
    
    sector_edges = np.linspace(0, 360, sectors + 1)
    sector_labels = ['N', 'NE', 'E', 'SE','S', 'SW', 'W', 'NW']
    
    data['wind_sector'] = pd.cut(
    adjusted_wd, 
    bins=sector_edges, 
    labels=sector_labels, 
    right=False, 
    include_lowest=True
    )

    # Calculate the frequency for each wind direction and keep only the frequencency >= 0,9 * max freq
    freq = data['wind_sector'].value_counts(normalize=True).sort_index()
    dominant_sectors = freq[freq>= 0.9 * max(freq)].index

    data_dominant = data[data['wind_sector'].isin(dominant_sectors)]

    print('The prevailing winds are: ' + ', '.join(dominant_sectors))
    print(freq)
    return data_dominant