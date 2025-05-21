import numpy as np
import pandas as pd

def flow_duration_slope(df=None, stream_col=None):

    # Slope of the Flow Duration Curve (FDC)
    """
    Calculate the slope of the Flow Duration Curve (FDC) using the exceedance probability method for each year
    and then average the results across all years.

    Parameters:
    df (DataFrame): A DataFrame with 'datetime' and streamflow column.
    stream_col (str): The column name for streamflow values.

    Returns:
    overall_slope (float): The average slope of the FDC across all years.
    yearly_slopes (dict): A dictionary with years as keys and the slope of the FDC for each year as values.
    """
    yearly_slopes = {}

    for year, group in df.groupby('year'):
        # Sort the streamflow data in descending order
        sorted_flows = group[stream_col].sort_values(ascending=False).reset_index(drop=True)
        sorted_flows.index += 1  # Rank the flows (1, 2, 3, ..., N)

        # Calculate the exceedance probability
        exceedance_probability = sorted_flows.index / (len(sorted_flows) + 1)
        sorted_flows = pd.DataFrame({
            'Streamflow': sorted_flows,
            'Exceedance_Probability': exceedance_probability
        })

        # Select two points to calculate the slope (33rd and 66th percentiles)
        Q33 = sorted_flows.loc[np.abs(sorted_flows['Exceedance_Probability'] - 0.33).idxmin(), 'Streamflow']
        P33 = sorted_flows.loc[np.abs(sorted_flows['Exceedance_Probability'] - 0.33).idxmin(), 'Exceedance_Probability']

        Q66 = sorted_flows.loc[np.abs(sorted_flows['Exceedance_Probability'] - 0.66).idxmin(), 'Streamflow']
        P66 = sorted_flows.loc[np.abs(sorted_flows['Exceedance_Probability'] - 0.66).idxmin(), 'Exceedance_Probability']

        # Calculate the slope of the FDC for the year
        slope = (Q33 - Q66) / (P33 - P66)
        yearly_slopes[year] = slope

    # Calculate the overall average slope across all years
    overall_slope = np.mean(list(yearly_slopes.values()))
    
    return overall_slope, yearly_slopes


def streamflow_precipitation_elasticity(df=None, stream_col=None): # OK
    yearly_elasticities = {}

    for year, group in df.groupby('year'):
        mean_precipitation = group['precipitation'].mean()
        mean_streamflow = group[stream_col].mean()

        delta_p = (group['precipitation'] - mean_precipitation) / mean_precipitation
        delta_q = (group[stream_col] - mean_streamflow) / mean_streamflow

        elasticity = (delta_q / delta_p).mean()
        yearly_elasticities[year] = elasticity

    overall_elasticity = np.mean(list(yearly_elasticities.values()))
    return overall_elasticity, yearly_elasticities

def frequency_of_high_flow_days(df=None, quantile=90, stream_col=None): # OK
    yearly_high_flow_days = {}

    for year, group in df.groupby('year'):
        threshold = np.percentile(group[stream_col], quantile)
        high_flow_days = (group[stream_col] > threshold).sum()
        yearly_high_flow_days[year] = high_flow_days

    overall_high_flow_frequency = np.mean(list(yearly_high_flow_days.values()))
    return overall_high_flow_frequency, yearly_high_flow_days

def mean_half_flow_date(df=None, stream_col=None): 
    yearly_half_flow_weeks = {}

    for year, group in df.groupby('year'):
        cumulative_flow = group[stream_col].cumsum()
        half_flow = cumulative_flow.max() / 2
        half_flow_date = group.loc[cumulative_flow >= half_flow, 'datetime'].iloc[0]
        
        # Get the week number of the half-flow date
        half_flow_week = half_flow_date.isocalendar()[1]  # Extract the ISO week number
        
        yearly_half_flow_weeks[year] = half_flow_week

    # Calculate the overall mean half-flow week
    overall_half_flow_week = int(np.mean(list(yearly_half_flow_weeks.values())))
    
    return overall_half_flow_week, yearly_half_flow_weeks

def average_duration_of_low_flow_events(df=None, quantile=10, stream_col=None):
    yearly_avg_durations = {}

    for year, group in df.groupby('year'):
        threshold = np.percentile(group[stream_col], quantile)
        low_flow_events = []
        current_duration = 0

        for flow in group[stream_col]:
            if flow < threshold:
                current_duration += 1
            else:
                if current_duration > 0:
                    low_flow_events.append(current_duration)
                    current_duration = 0

        if current_duration > 0:
            low_flow_events.append(current_duration)

        yearly_avg_durations[year] = np.mean(low_flow_events) if low_flow_events else 0

    overall_avg_duration = np.mean(list(yearly_avg_durations.values()))
    return overall_avg_duration, yearly_avg_durations

def frequency_of_zero_flow_days(df=None, stream_col=None): # check again
    """
    Calculate the frequency of days with zero flow for each year and overall average.

    Parameters:
    df (DataFrame): A DataFrame with 'datetime' and streamflow column.
    stream_col (str): The column name for streamflow values.

    Returns:
    overall_zero_flow_frequency (float): The average number of zero flow days per year.
    yearly_zero_flow_days (dict): A dictionary with years as keys and the number of zero flow days for each year as values.
    """
    yearly_zero_flow_days = {}

    for year, group in df.groupby('year'):
        zero_flow_days = (group[stream_col] <= 1).sum()
        yearly_zero_flow_days[year] = zero_flow_days

    overall_zero_flow_frequency = np.mean(list(yearly_zero_flow_days.values()))
    
    return overall_zero_flow_frequency, yearly_zero_flow_days

def calculate_runoff_ratio(df, stream_col=None, precipitation_col='precipitation'):
    """
    Calculate the runoff ratio for each year and overall average.

    Parameters:
    df (DataFrame): A DataFrame with 'datetime', streamflow, and total_precipitation columns.
    stream_col (str): The column name for streamflow values.
    total_precipitation_col (str): The column name for total precipitation values.

    Returns:
    overall_runoff_ratio (float): The average runoff ratio across all years.
    yearly_runoff_ratios (dict): A dictionary with years as keys and the runoff ratio for each year as values.
    """
    yearly_runoff_ratios = {}

    for year, group in df.groupby('year'):
        total_runoff = group[stream_col].sum()
        total_precipitation = group[precipitation_col].sum()

        if total_precipitation == 0:
            print(f"Warning: Total precipitation is zero for year {year}. Skipping calculation.")
            continue
        
        runoff_ratio = total_runoff / total_precipitation
        yearly_runoff_ratios[year] = runoff_ratio

    overall_runoff_ratio = np.mean(list(yearly_runoff_ratios.values()))
    
    return overall_runoff_ratio, yearly_runoff_ratios

def calculate_5_percent_flow_quantile(df=None, stream_col=None):
    """
    Calculate the 5% flow quantile for each year and overall average.

    Parameters:
    df (DataFrame): A DataFrame with 'datetime' and streamflow column.
    stream_col (str): The column name for streamflow values.

    Returns:
    overall_5_percent_quantile (float): The average 5% flow quantile across all years.
    yearly_quantiles (dict): A dictionary with years as keys and the 5% flow quantile for each year as values.
    """
    yearly_quantiles = {}

    for year, group in df.groupby('year'):
        flow_quantile_5 = np.percentile(group[stream_col], 5)
        yearly_quantiles[year] = flow_quantile_5

    overall_5_percent_quantile = np.mean(list(yearly_quantiles.values()))
    
    return overall_5_percent_quantile, yearly_quantiles


def calculate_95_percent_flow_quantile(df=None, stream_col=None):
    """
    Calculate the 95% flow quantile for each year and overall average.

    Parameters:
    df (DataFrame): A DataFrame with 'datetime' and streamflow column.
    stream_col (str): The column name for streamflow values.

    Returns:
    overall_95_percent_quantile (float): The average 95% flow quantile across all years.
    yearly_quantiles (dict): A dictionary with years as keys and the 95% flow quantile for each year as values.
    """
    yearly_quantiles = {}

    for year, group in df.groupby('year'):
        flow_quantile_95 = np.percentile(group[stream_col], 95)
        yearly_quantiles[year] = flow_quantile_95

    overall_95_percent_quantile = np.mean(list(yearly_quantiles.values()))
    
    return overall_95_percent_quantile, yearly_quantiles

def frequency_of_low_flow_days(df, stream_col=None, quantile=10):
    """
    Calculate the frequency of low flow days for each year and overall average using a specified quantile as the threshold.

    Parameters:
    df (DataFrame): A DataFrame with 'datetime' and streamflow column.
    stream_col (str): The column name for streamflow values.
    quantile (float): The quantile to be used for determining the low flow threshold (default is 10%).

    Returns:
    overall_low_flow_frequency (float): The average number of low flow days per year.
    yearly_low_flow_days (dict): A dictionary with years as keys and the number of low flow days for each year as values.
    """
    yearly_low_flow_days = {}

    for year, group in df.groupby('year'):
        threshold = np.percentile(group[stream_col], quantile)
        low_flow_days = (group[stream_col] < threshold).sum()
        yearly_low_flow_days[year] = low_flow_days

    overall_low_flow_frequency = np.mean(list(yearly_low_flow_days.values()))
    
    return overall_low_flow_frequency, yearly_low_flow_days

import numpy as np
import pandas as pd

def eckhardt_filter(streamflow=None, alpha=0.98, BFI_max=0.8):
    """
    Apply the Eckhardt recursive digital filter to estimate baseflow.

    Parameters:
    streamflow (array-like): Array or list of streamflow values.
    alpha (float): Recession constant (typically between 0.95 and 0.99).
    BFI_max (float): Maximum baseflow index (typically between 0.5 and 0.8).

    Returns:
    baseflow (np.array): Estimated baseflow values.
    """
    baseflow = np.zeros(len(streamflow))
    baseflow[0] = streamflow[0]  # Initial baseflow is set to the first streamflow value
    
    for i in range(1, len(streamflow)):
        baseflow[i] = ((1 - BFI_max) * alpha * baseflow[i-1] + (1 - alpha) * BFI_max * streamflow[i]) / (1 - alpha * BFI_max)
    
    return baseflow

def baseflow_index(df, stream_col=None, alpha=0.98, BFI_max=0.7):
    """
    Calculate the Baseflow Index (BFI) using the Eckhardt filter method for each year and then average the results.

    Parameters:
    df (DataFrame): A DataFrame with 'datetime' and streamflow column.
    stream_col (str): The column name for streamflow values.
    alpha (float): Recession constant for the Eckhardt filter.
    BFI_max (float): Maximum baseflow index for the Eckhardt filter.

    Returns:
    overall_bfi (float): The average Baseflow Index (BFI) across all years.
    yearly_bfi (dict): A dictionary with years as keys and the BFI for each year as values.
    """
    yearly_bfi = {}

    for year, group in df.groupby('year'):
        streamflow = group[stream_col].values
        
        # Calculate the baseflow using the Eckhardt filter method
        baseflow = eckhardt_filter(streamflow, alpha=alpha, BFI_max=BFI_max)
        
        # Calculate the total baseflow and total streamflow for the year
        total_baseflow = np.sum(baseflow)
        total_streamflow = np.sum(streamflow)
        
        # Prevent division by zero or very small streamflow values
        if total_streamflow == 0:
            print(f"Warning: Total streamflow is zero for year {year}. Skipping calculation.")
            continue
        
        # Calculate the Baseflow Index (BFI) for the year
        bfi = total_baseflow / total_streamflow
        yearly_bfi[year] = bfi

    # Calculate the overall average BFI across all years
    overall_bfi = np.mean(list(yearly_bfi.values()))
    
    return overall_bfi, yearly_bfi

