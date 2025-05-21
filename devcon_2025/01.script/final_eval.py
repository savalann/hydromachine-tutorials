from evaluation_table import EvalTable
import matplotlib.pyplot as plt
import pandas as pd
import math
from signatures import *
import dataretrieval.nwis as nwis
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.axisartist import Axes
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


def general_viz(df_result_data, station_list, type):
    if type == 0:
        # Calculate the number of subplots needed based on the number of unique stations.
        n_subplots = len(station_list)
        # Determine the number of columns in the subplot grid by taking the ceiling of the square root of 'n_subplots'.
        n_cols = 1 #int(math.ceil(math.sqrt(n_subplots)))
        # Determine the number of rows in the subplot grid by dividing 'n_subplots' by 'n_cols' and taking the ceiling of that.
        n_rows = 3#int(math.ceil(n_subplots / n_cols))
        # Set the figure size for the subplots.
        figsize = (10, 12)
        # Create a grid of subplots with specified number of rows and columns and figure size.
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        # Flatten the axes array for easier iteration.
        axes = axes.flatten()
        
        # Iterate over the axes to plot the data for each station.
        for i, ax in enumerate(axes):
            if i < n_subplots:
                # Extract the data for the current station from the dataset.
                temp_df_1 = df_result_data[station_list[i]]
                # Set 'datetime' as the index for plotting.
                temp_df_2 = temp_df_1.set_index('datetime')
                # Plot the 'flow_cfs' data on the primary y-axis.
                ax.plot(temp_df_2.index, temp_df_2['flow_cms'], label='Observation')
                # Set the x-axis limits from the first to the last year of data.
                start_year = pd.to_datetime(f'{temp_df_1.datetime.dt.year.min()}-01-01')
                end_year = pd.to_datetime(f'{temp_df_1.datetime.dt.year.max()}-12-31')
                ax.set_xlim(start_year, end_year)
                # Rotate x-axis labels for better readability.
                labels = ax.get_xticklabels()
                ax.set_xticklabels(labels, rotation=45)
        
                # Extract the data for the current station from the dataset.
                temp_df_1_lstm = df_result_data[station_list[i]]
                # Set 'datetime' as the index for plotting.
                temp_df_2_lstm = temp_df_1_lstm.set_index('datetime')
                # Plot the 'flow_cfs' data on the primary y-axis.
                ax.plot(temp_df_2_lstm.index, temp_df_2_lstm['lstm_flow'], label='LSTM')
                # Set the x-axis limits from the first to the last year of data.
                start_year = pd.to_datetime(f'{temp_df_1_lstm.datetime.dt.year.min()}-01-01')
                end_year = pd.to_datetime(f'{temp_df_1_lstm.datetime.dt.year.max()}-12-31')
                ax.set_xlim(start_year, end_year)
                # Rotate x-axis labels for better readability.
                labels = ax.get_xticklabels()
                ax.set_xticklabels(labels, rotation=45)
        
                # Extract the data for the current station from the dataset.
                temp_df_1_nwm = df_result_data[station_list[i]]
                # Set 'datetime' as the index for plotting.
                temp_df_2_nwm = temp_df_1_nwm.set_index('datetime')
                # Plot the 'flow_cfs' data on the primary y-axis.
                ax.plot(temp_df_2_nwm.index, temp_df_2_nwm['NWM_flow'], label="NWM")
                # Set the x-axis limits from the first to the last year of data.
                start_year = pd.to_datetime(f'{temp_df_1_nwm.datetime.dt.year.min()}-01-01')
                end_year = pd.to_datetime(f'{temp_df_1_nwm.datetime.dt.year.max()}-12-31')
                ax.set_xlim(start_year, end_year)
                # Rotate x-axis labels for better readability.
                labels = ax.get_xticklabels()
                ax.set_xticklabels(labels, rotation=45)
                ax.legend()
                
                # Set the title of the subplot to the station ID.
                ax.set_title(f'{station_list[i]}')
                # Set the x-axis label for subplots in the last row.
                if i // n_cols == n_rows - 1:
                    ax.set_xlabel('Datetime (day)')
        
                # Set the y-axis label for subplots in the first column.
                if i % n_cols == 0:
                    ax.set_ylabel('Streamflow (cfs)')
            else:
                # Hide any unused axes.
                ax.axis('off')
            
        # Adjust layout to prevent overlapping elements.
        plt.tight_layout()
        # Uncomment the line below to save the figure to a file.
        # plt.savefig(f'{save_path}scatter_annual_drought_number.png')
        # Display the plot.
        plt.show()

    if type == 1:
        # Calculate the number of subplots needed based on the number of unique stations.
        n_subplots = len(station_list)
        # Determine the number of columns in the subplot grid by taking the ceiling of the square root of 'n_subplots'.
        n_cols = 3# int(math.ceil(math.sqrt(n_subplots)))
        # Determine the number of rows in the subplot grid by dividing 'n_subplots' by 'n_cols' and taking the ceiling of that.
        n_rows = int(math.ceil(n_subplots / n_cols))
        # Set the figure size for the subplots.
        figsize = (24,8)
        # Create a grid of subplots with specified number of rows and columns and figure size.
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        # Flatten the axes array for easier iteration.
        axes = axes.flatten()
        
        # Iterate over the axes to plot the data for each station.
        for i, ax in enumerate(axes):
            if i < n_subplots:
                # Extract the data for the current station from the dataset.
                temp_df_1 = df_result_data[station_list[i]]
                obs = temp_df_1['flow_cms']
                model_1 = temp_df_1['lstm_flow']
                model_2 = temp_df_1['NWM_flow']
                
                # Create the scatter plot
                ax.scatter(obs, model_1, alpha=0.7, label='LSTM')
                ax.scatter(obs, model_2, alpha=0.5, label='NWM')
                
                # Plot 1:1 line
                min_val = min(obs.min(), model_1.min(), model_2.min())
                max_val = max(obs.max(), model_1.max(), model_2.min())
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
        
                ax.legend()
                
                # Set the title of the subplot to the station ID.
                ax.set_title(f'{station_list[i]}')
                # Set the x-axis label for subplots in the last row.
                if i // n_cols == n_rows - 1:
                    ax.set_xlabel('Observation (cms)')
        
                # Set the y-axis label for subplots in the first column.
                if i % n_cols == 0:
                    ax.set_ylabel('Model (cms)')
            else:
                # Hide any unused axes.
                ax.axis('off')
            
        # Adjust layout to prevent overlapping elements.
        plt.tight_layout()
        # Uncomment the line below to save the figure to a file.
        # plt.savefig(f'{save_path}scatter_annual_drought_number.png')
        # Display the plot.
        plt.show()


def regime_eval(df_result_data, station_list):
    
    # Function to filter rows based on decision
    def filter_streamflow(df, decision):
        percentile_20 = df['flow_cms'].quantile(0.20)
        percentile_80 = df['flow_cms'].quantile(0.80)
        if decision == 'low':
            # Rows below the 20th percentile
            return df[df['flow_cms'] < percentile_20]
        elif decision == 'high':
            # Rows above the 80th percentile
            return df[df['flow_cms'] > percentile_80]
        elif decision == 'normal':
            # Rows between the 20th and 80th percentiles
            return df[(df['flow_cms'] >= percentile_20) & (df['flow_cms'] <= percentile_80)]
        else:
            return df  # No filtering if invalid decision
    
    # Example usage:
    data_list = []
    for decision in ['low', 'normal', 'high']:  # Can be 'below_20', 'above_80', or 'between_20_80'
        EvalDF_all_rf_regime = pd.DataFrame()
        SupplyEvalDF_all_rf_regime = pd.DataFrame()
        df_eval_rf_regime = pd.DataFrame()
        df_result_data_regime = {}
    
        # Iterate over each station name in the list of station IDs.
        for station_name in station_list:
        
            filtered_df = filter_streamflow(df_result_data[station_name], decision)
       
            # Assuming EvalTable is a predefined function that compares predictions to actuals and returns evaluation DataFrames.
            EvalDF_all_rf_temp_regime, SupplyEvalDF_all_rf_temp_regime, df_eval_rf_temp_regime = EvalTable(filtered_df['lstm_flow'].to_numpy().reshape(-1), filtered_df[['station_id', 'NHDPlusid', 'datetime', 'state', 'dec_lat_va',
               'dec_long_va', 'total_area', 'basin_mean_elevation', 'basin_mean_slope',
               'imperv_perc', 'agri_perc', 'forest_perc', 'day_of_year', 's1', 's2',
               'precipitation', 'temperature', 'storage', 'swe', 'NWM_flow', 'min',
               'max', 'mean', 'median', 'flow_cms', 'lstm_flow']], 'LSTM')
     
            # Append the results from each station to the respective DataFrame.
            EvalDF_all_rf_regime = pd.concat([EvalDF_all_rf_regime, EvalDF_all_rf_temp_regime], ignore_index=True)
            SupplyEvalDF_all_rf_regime = pd.concat([SupplyEvalDF_all_rf_regime, SupplyEvalDF_all_rf_temp_regime], ignore_index=True)
            df_eval_rf_regime = pd.concat([df_eval_rf_regime, df_eval_rf_temp_regime], ignore_index=True)
        EvalDF_all_rf_regime['flow_type'] = decision
        data_list.append(EvalDF_all_rf_regime)
        
        # print(f"Model performance for {decision} flows")
        # print('')
        # display(EvalDF_all_rf_regime[['USGSid','NWM_KGE', 'LSTM_KGE']])   
    
    
    data = pd.concat(data_list, ignore_index=True)
    display(data)



def signature_eval(df_result_data, station_list, output_path):
    
    
    human_station = ['10146000', '10146400', '10155000', '10166430', '10171000', '10039500', '10068500', '10092700']
    
    
    headwater_station = ['10011500', '10105900', '10109000', '10113500', '10128500', '10131000', '10137500', '10145400', '10150500', '10154200']
    
    
    reservoir_station = ['10126000', '10129500', '10130500', '10132000', '10132500', '10134500', '10136500', '10140100', '10141000', '10155500', '10156000', '10168000']
    
    
    data_station_signature = {}
    signature_function = [flow_duration_slope, 
                          streamflow_precipitation_elasticity, 
                          frequency_of_high_flow_days, 
                          mean_half_flow_date, 
                          average_duration_of_low_flow_events,
                          frequency_of_zero_flow_days, 
                          calculate_runoff_ratio, 
                          calculate_5_percent_flow_quantile, 
                          calculate_95_percent_flow_quantile, 
                          frequency_of_low_flow_days, 
                          baseflow_index
                         ] 
    
    functions_list_str = [
        'flow_duration_slope', 
        'streamflow_precipitation_elasticity', 
        'frequency_of_high_flow_days', 
        'mean_half_flow_date', 
        'average_duration_of_low_flow_events',
        'frequency_of_zero_flow_days', 
        'calculate_runoff_ratio', 
        'calculate_5_percent_flow_quantile', 
        'calculate_95_percent_flow_quantile', 
        'frequency_of_low_flow_days', 
        'baseflow_index'
    ]
    
    key_list = list(df_result_data.keys())
        
    for signature_index, signature_name in enumerate(functions_list_str):
    
        data_signatures = np.zeros((len(key_list), 3))
        
        for model_index, model_name in enumerate(['flow_cms', 'NWM_flow', 'lstm_flow']):
    
            for station_index, station_num in enumerate(key_list):
    
                temp_df_1 = df_result_data[station_num].copy()
    
                temp_df_1.rename(columns={'Datetime': 'datetime'}, inplace=True)
                
                temp_df_1['year'] = temp_df_1['datetime'].dt.year
    
                # Dynamically access the function from the si module using getattr
                function_to_call = signature_function[signature_index]
                data_signatures[station_index, model_index] = function_to_call(df=temp_df_1, stream_col=model_name)[0]
                
        data_station_signature[signature_name] = data_signatures
    
    result_df = {}
    
    
    final_signature = np.zeros((len(functions_list_str), 2))
    
    for signature_index, signature_name in enumerate(functions_list_str):
    
        original_array = data_station_signature[signature_name]
        
        # Filter out zero values in the original array to avoid division by zero
        mask_nonzero = original_array[:, 0] != 0
        
        # Calculate MAPE values only for non-zero observed values
        mape_2_1 = np.mean(np.abs((original_array[mask_nonzero, 0] - original_array[mask_nonzero, 1]) / original_array[mask_nonzero, 0])) * 100
        mape_3_1 = np.mean(np.abs((original_array[mask_nonzero, 0] - original_array[mask_nonzero, 2]) / original_array[mask_nonzero, 0])) * 100
        
        # Store the MAPE values in the final_signature array
        final_signature[signature_index, 0] = round(mape_2_1, 4)
        final_signature[signature_index, 1] = round(mape_3_1, 4)
        
    # Convert to DataFrame for display
    result_df['all'] = pd.DataFrame(final_signature, columns=['NWM', 'LSTM'], index=functions_list_str)
        
    # # Display the resulting DataFrame
    # result_df['all']
    # Create a DataFrame (assuming 'result_df' is already defined)
    df = result_df['all']
    
    # Custom x-axis labels
    custom_labels = [
        "Flow Duration Slope",                 
        "Streamflow Precipitation Elasticity", 
        "Frequency of High Flow Days",         
        "Mean Half Flow Date",                 
        "Average Duration of Low Flow Events", 
        "Frequency of Zero Flow Days",         
        "Runoff Ratio",                        
        "5% Flow Quantile",                    
        "95% Flow Quantile",                   
        "Frequency of Low Flow Days",          
        "Baseflow Index"                       
    ]
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(25, 20))
    
    # Create bar width and x positions
    bar_width = 0.35
    index = np.arange(len(df))
    
    # Clip the y values to 100 for plotting
    clipped_df = df.clip(upper=100)
    
    # Plot bars for each column
    bar1 = ax.bar(index, clipped_df['NWM'], bar_width, label='NWM')
    bar2 = ax.bar(index + bar_width, clipped_df['LSTM'], bar_width, label='PP-ML')
    
    # Set x-ticks and custom labels
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(custom_labels, rotation=45, ha='right', fontsize=22)
    
    # Set labels and title
    ax.set_xlabel("Hydrological Signatures", fontsize=24)
    ax.set_ylabel("MAPE", fontsize=24)
    
    # Set y-axis limit and custom tick labels without 110
    ax.set_ylim(0, 123)
    ax.set_yticks([0, 20, 40, 60, 80, 100])  # Specify y-ticks without including 110
    
    # Increase font size of x and y-axis tick labels
    ax.tick_params(axis='x', labelsize=22)  # Increase x-axis tick label size
    ax.tick_params(axis='y', labelsize=22)  # Increase y-axis tick label size
    
    # Annotate bars where the original value was above 100
    for i, value in enumerate(df['NWM']):
        if value > 100:
            ax.text(i, clipped_df['NWM'][i] + 2, int(round(value, 0)), ha='center', va='bottom', color='red', fontsize=20, rotation=45)
    
    for i, value in enumerate(df['LSTM']):
        if value > 100:
            ax.text(i + bar_width, clipped_df['LSTM'][i] + 2, int(round(value, 0)), ha='center', va='bottom', color='red', fontsize=20, rotation=45)
    
    # Show the legend
    plt.legend(fontsize=22)
    
    # Show the plot
    plt.tight_layout()  # Ensures the labels fit nicely
    plt.savefig(f'{output_path}signatures_all.png', bbox_inches='tight')
    plt.show()




def eval_drought(df_result_data, station_list, duration, dataset, EvalDF_all_rf):
    def sdf_creator(status='all', site='', data='', duration='all', figure=True, threshold=None):

        sufficient_year_number = [30]  # The least number of years which is sufficient for SDF curve creation.

        if len(site):  # Checks to see whether the user asks for USGS gage or a file related SDF curve.

            raw_data = Pyndat.daily_data(site=site)  # Calls the function to get the daily USGS gage data.

            if raw_data.shape[1] != 6 or raw_data.shape[1] == 0:  # Checks to whether data is ok or not.

                sys.exit(1)

        elif len(data):

            data = data.rename({'Flow': 'USGS_flow'}, axis=1)

            data['Datetime'] = pd.to_datetime(data['Datetime'])

            raw_data = data

        year_range = raw_data['Datetime'].dt.year.unique()  # Finds the years of the time-series.

        max_year = year_range[-1]  # The last year of the time-series.

        min_year = year_range[0]  # The first year of the time-series.

        max_month = raw_data.iloc[-1, 0].month  # The first month of the time-series.

        min_month = raw_data.iloc[0, 0].month  # The first month of the time-series.

        if max_month > 10:  # Checks to see whether the last year has the complete water year data.

            raw_data = raw_data.loc[(raw_data['Datetime'] < str(max_year) + '-10-01')]

        else:

            raw_data = raw_data.loc[(raw_data['Datetime'] < (str(year_range[-2]) + '-10-01'))]

            max_year = year_range[-2]  # If last year is not complete (does not have October), it goes one year back.

        if min_month < 10:  # Checks to see whether the first year has the complete water year data.

            raw_data = raw_data.loc[(raw_data['Datetime'] >= (str(min_year) + '-10-01'))]

        else:

            raw_data = raw_data.loc[(raw_data['Datetime'] >= (str(year_range[1]) + '-10-01'))]

            min_year = year_range[1]  # If first year is not complete (does not have October), it goes one year forward.

        if status == 'optimal':  # Checks whether the user wants the SDF curve based on the best data criteria. *

            # Whether data is more than the least number of required years.
            if len(year_range) < sufficient_year_number[0]:

                print('Lack of Sufficient Number of Years Warning: The number of years is less than sufficient.')

            # Whether it has the sufficient recent years.
            elif not any(x in year_range for x in list(range(datetime.today().year - sufficient_year_number[0] - 1,
                                                             datetime.today().year + 1))):
                print('Lack of Most Recent Data Warning: The data does not contain the last recent years.')

            else:
                min_year = year_range[-sufficient_year_number[0]]  # Determines the first of the updated time-series.

                # Uses the minimum number of years needed.
                raw_data = raw_data[(raw_data['Datetime'] >= (str(min_year) + '-10-01'))]

                year_range = raw_data['Datetime'].dt.year.unique()  # Updates the years list of time-series.

        mean_year = np.zeros((len(year_range) - 1, 5))

        for ii in range(len(mean_year)):  # Writes the year and the average value of it.

            mean_year[ii, 0] = int(year_range[ii + 1])  # Year number.

            mean_year[ii, 1] = np.median(
                raw_data[(raw_data['Datetime'] >= (str(year_range[ii]) + '-10-01')) &  # Mean of that year.
                         (raw_data['Datetime'] < (str(year_range[ii + 1]) + '-10-01')) &
                         (raw_data['USGS_flow'] >= 0)]['USGS_flow'])  # Neglects the negative data.

        mean_year = mean_year[~(np.isnan(mean_year)).any(axis=1)]  # Remove the NANs.
        
        if threshold != None:

            overall_average = threshod
        else:
            overall_average = np.median(mean_year[:, 1])  # The average of the whole data set.

            threshod = overall_average
            
        if duration == 'all':  # Checks which duration user asked for.

            duration = list(range(2, 11))

        else:

            duration = list(map(int, duration.split(',')))

        # Pre-defined colors for the SDF curve.
        color = ['b', 'g', 'y', 'r', 'orange', 'brown', 'gray', 'cyan', 'olive', 'pink']

        # Calculates the SDF curve for each duration.
        for dd in range(len(duration)):

            arrays = [['Duration=' + str(duration[dd])] * 5,
                      ["Date", "Mean_Flow(cfs)", 'Rolling_Average(cfs)', 'Severity(%)',
                       'Probability(%)']]

            tuples = list(zip(*arrays))

            index = pd.MultiIndex.from_tuples(tuples)

            mean_year_temp = pd.DataFrame(mean_year)

            # Calculates the moving average of required duration.
            mean_year_temp.iloc[:, 2] = mean_year_temp.iloc[:, 1].rolling(duration[dd]).mean()

            mean_year_temp = mean_year_temp.dropna()  # Remove the NANs of each moving average.

            # Calculates the severity.
            mean_year_temp.iloc[:, 3] = (mean_year_temp.iloc[:, 2] - overall_average) / overall_average * 100

            # Get all the wet and drought severity values
            final_all_years_temp = pd.DataFrame(mean_year_temp.to_numpy(), columns=index).iloc[:, :-1]

            mean_year_temp = mean_year_temp[mean_year_temp.iloc[:, 3] <= 0]  # Removes the non drought severity.

            # Sort the data for frequency calculation
            temp_severity = mean_year_temp.sort_values(by=3, axis=0, ascending=False)

            for i in range(len(mean_year_temp)):  # Calculates the non-exceedance probability.

                temp_severity.iloc[i, 4] = (i + 1) / len(mean_year_temp) * 100

            temp_severity = temp_severity.sort_values(by=0, axis=0)

            temp_final = pd.DataFrame(temp_severity.to_numpy(), columns=index)

            if dd == 0:

                final = temp_final.reset_index(drop=True)
                final_all_years = final_all_years_temp.reset_index(drop=True)

            else:

                final = pd.concat([final.reset_index(drop=True),
                                   temp_final.reset_index(drop=True)], axis=1)
                final_all_years = pd.concat([final_all_years.reset_index(drop=True),
                                   final_all_years_temp.reset_index(drop=True)], axis=1)

        # Plotting the SDF curve.
        fig = 0

        # plt.rcParams["font.family"] = "Times New Roman"

        fig = plt.figure(dpi=300, layout="constrained", facecolor='whitesmoke')

        axs = fig.add_subplot(axes_class=Axes, facecolor='whitesmoke')

        axs.axis["right"].set_visible(False)

        axs.axis["top"].set_visible(False)

        axs.axis["left"].set_axisline_style("-|>")

        axs.axis["left"].line.set_facecolor("black")

        axs.axis["bottom"].set_axisline_style("-|>")

        axs.axis["bottom"].line.set_facecolor("black")

        plt.title(label='SDF Curve', fontsize=20, pad=10)

        axs.axis["bottom"].label.set_text("Severity (%)")

        axs.axis["bottom"].label.set_fontsize(15)

        axs.axis["left"].label.set_text("Non-Exceedance Probability")

        axs.axis["left"].label.set_fontsize(15)

        for dd, ii in enumerate(duration):
            filled_marker_style = dict(marker='o', linestyle='-', markersize=5,
                                       color=color[dd])

            temp_final = final[('Duration=' + str(ii))].sort_values(by=['Probability(%)'])

            axs.plot(temp_final.iloc[:, 3] * (-1), temp_final.iloc[:, 4], **filled_marker_style,
                     label=('Duration = ' + str(ii)))

        plt.legend(loc='lower right')

        if figure is True:

            plt.show()

        elif figure is False:

            plt.close()

        return final, raw_data, fig, final_all_years, threshold

    dataset.rename(columns={'dec_lat_va': 'Lat', 'dec_long_va': 'Long'}, inplace=True)
    EvalDF_all_rf.rename(columns={'USGSid': 'station_id'}, inplace=True)
    df_modified = dataset[['station_id', 'Lat', 'Long']]
    df_modified = dataset[['station_id', 'Lat', 'Long']].drop_duplicates().reset_index(drop=True)
    EvalDF_all_rf_all = pd.merge(EvalDF_all_rf, df_modified[['station_id', 'Lat', 'Long']], on='station_id')


    # Iterate over each station name in the list of station IDs.
    
    drought_data = pd.DataFrame([], columns=['station_id', 'NWM', 'LSTM'])
    drought_data['station_id'] = EvalDF_all_rf_all['station_id']
    
    
    drought_time = pd.DataFrame([], columns=['station_id', 'NWM', 'LSTM'])
    drought_time['station_id'] = EvalDF_all_rf_all['station_id']
    
    
    for station_index, station_name in enumerate(station_list):
        temp_data_1 = None
        temp_data_1 = df_result_data[station_name] 
        temp_data_obs = temp_data_1[['datetime', 'flow_cms']].reset_index(drop=True)
        temp_data_nwm = temp_data_1[['datetime', 'NWM_flow']].reset_index(drop=True)
        temp_data_xgb = temp_data_1[['datetime', 'lstm_flow']].reset_index(drop=True)
    
    
        temp_data_obs = temp_data_obs.rename(columns={'datetime': 'Datetime', 'flow_cms': 'USGS_flow'}).sort_values(by='Datetime').reset_index(drop=True)
        temp_data_nwm = temp_data_nwm.rename(columns={'datetime': 'Datetime', 'NWM_flow': 'USGS_flow'}).sort_values(by='Datetime').reset_index(drop=True)
        temp_data_xgb = temp_data_xgb.rename(columns={'datetime': 'Datetime', 'lstm_flow': 'USGS_flow'}).sort_values(by='Datetime').reset_index(drop=True) 
    
        
        _, _, _, drought_obs, threshold_obs = sdf_creator(data=temp_data_obs, figure=False, duration='2, 5')
        _, _, _, drought_nwm, _ = sdf_creator(data=temp_data_nwm, figure=False, duration='2, 5', threshold=threshold_obs)
        _, _, _, drought_xgb, _  = sdf_creator(data=temp_data_xgb, figure=False, duration='2, 5', threshold=threshold_obs)
    
        # drought_data[station_index, 0] = len(drought_obs['Duration=2']['Severity(%)'])#.median()
        # drought_data[station_index, 1] = len(drought_nwm['Duration=2']['Severity(%)'])#.median()
        # drought_data[station_index, 2] = len(drought_xgb['Duration=2']['Severity(%)'])#.median()
     
        drought_data.iloc[station_index, 1] = round(r2_score(drought_obs[f'Duration={duration}']['Severity(%)'].dropna(), drought_nwm[f'Duration={duration}']['Severity(%)'].dropna()), 2)
        
        drought_data.iloc[station_index, 2] = round(r2_score(drought_obs[f'Duration={duration}']['Severity(%)'].dropna(), drought_xgb[f'Duration={duration}']['Severity(%)'].dropna()), 2)
    
    
        drought_obs, _, _, _, threshold_obs = sdf_creator(data=temp_data_obs, figure=False, duration='2, 5')
        drought_nwm, _, _, _, _ = sdf_creator(data=temp_data_nwm, figure=False, duration='2, 5', threshold=threshold_obs)
        drought_xgb, _, _, _, _ = sdf_creator(data=temp_data_xgb, figure=False, duration='2, 5', threshold=threshold_obs)
    
        drought_time.iloc[station_index, 1] = len(drought_nwm[f'Duration={duration}']['Date'].isin(drought_obs[f'Duration={duration}']['Date'])) / len(drought_obs[f'Duration={duration}']['Date'])
    
        drought_time.iloc[station_index, 2] = len(drought_xgb[f'Duration={duration}']['Date'].isin(drought_obs[f'Duration={duration}']['Date'])) / len(drought_obs[f'Duration={duration}']['Date'])
    
    display(drought_data)


# import matplotlib.pyplot as plt
# import pandas as pd

# # Recreate your data
# # data = pd.DataFrame({
# #     'USGSid': [10154200, 10155000, 10156000] * 3,
# #     'NWM_KGE': [-2.65, -3.04, -0.18, 0.4, 0.45, -0.46, 0.6, 0.4, -0.03],
# #     'LSTM_KGE': [0.15, 0.11, -2.36, 0.44, 0.44, -1.22, 0.74, 0.67, -0.59],
# #     'flow_type': ['low']*3 + ['normal']*3 + ['high']*3
# # })

# # Define color per flow type
# flow_colors = {'low': 'red', 'normal': 'orange', 'high': 'green'}

# # Define marker per station
# station_ids = data['USGSid'].unique()
# station_markers = ['o', 's', '^']  # Add more if needed
# marker_map = dict(zip(station_ids, station_markers))

# # Plot
# fig, ax = plt.subplots(figsize=(8, 6))

# for _, row in data.iterrows():
#     ax.scatter(
#         row['NWM_KGE'],
#         row['LSTM_KGE'],
#         color=flow_colors[row['flow_type']],
#         marker=marker_map[row['USGSid']],
#         edgecolor='black',  # optional for contrast
#         s=100,
#         label=f"{row['flow_type']}_{row['USGSid']}"
#     )

# # Plot 1:1 reference line
# lims = [
#     min(data['NWM_KGE'].min(), data['LSTM_KGE'].min()) - 0.5,
#     max(data['NWM_KGE'].max(), data['LSTM_KGE'].max()) + 0.5,
# ]
# ax.plot(lims, lims, 'k--', label='1:1 Line')
# ax.set_xlim(lims)
# ax.set_ylim(lims)

# # Axis labels and title
# ax.set_xlabel('NWM KGE')
# ax.set_ylabel('LSTM KGE')
# ax.set_title('LSTM vs NWM performance by flow type and station')

# # Build custom legends
# # Legend for flow type (color)
# flow_legend = [plt.Line2D([0], [0], marker='o', color='w', label=ftype,
#                           markerfacecolor=color, markersize=10)
#                for ftype, color in flow_colors.items()]

# # Legend for stations (shape)
# station_legend = [plt.Line2D([0], [0], marker=marker, color='k', label=str(sid),
#                              markerfacecolor='w', markersize=10)
#                   for sid, marker in marker_map.items()]

# legend1 = ax.legend(handles=flow_legend, title="Flow Type", loc='upper left')
# legend2 = ax.legend(handles=station_legend, title="Station ID", loc='lower right')
# ax.add_artist(legend1)  # Add first legend manually to keep both

# plt.grid(False)
# plt.tight_layout()
# plt.show()
