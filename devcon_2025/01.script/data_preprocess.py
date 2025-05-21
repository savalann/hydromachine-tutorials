from sklearn.preprocessing import StandardScaler
import numpy as np 


def add_properties(train=None, test=None):


    # Group by station and day of the year, then calculate min, max, mean, and median
    aggregated_data = train.groupby(['station_id', 'day_of_year'])['flow_cms'].agg(['min', 'max', 'mean', 'median']).reset_index()

    # Merge the aggregated data back into the original DataFrame
    train_new = train.merge(aggregated_data, on=['station_id', 'day_of_year'], how='left', suffixes=('', '_agg'))

    train_new = train_new[['station_id', 'NHDPlusid', 'datetime', 'state', 'dec_lat_va',
       'dec_long_va', 'total_area', 'basin_mean_elevation', 'basin_mean_slope',
       'imperv_perc', 'agri_perc', 'forest_perc', 'day_of_year', 's1', 's2',
       'precipitation', 'temperature', 'storage', 'swe', 'NWM_flow', 'min', 'max', 'mean', 'median',
       'flow_cms']]

    # Now, use these parameters to modify the new dataset
    test_new = test.merge(aggregated_data, on=['station_id', 'day_of_year'], how='left', suffixes=('', '_agg'))

    test_new = test_new[['station_id', 'NHDPlusid', 'datetime', 'state', 'dec_lat_va',
       'dec_long_va', 'total_area', 'basin_mean_elevation', 'basin_mean_slope',
       'imperv_perc', 'agri_perc', 'forest_perc', 'day_of_year', 's1', 's2',
       'precipitation', 'temperature', 'storage', 'swe', 'NWM_flow', 'min', 'max', 'mean', 'median',
       'flow_cms']]

    return train_new, test_new

# Lookback creator
def add_lookback(daya_x_input, data_y_input, length_lookback):
    data_x = []
    data_y = []
    for i in range(len(daya_x_input)-length_lookback):
        features, targets = daya_x_input[i:i+length_lookback, :], data_y_input[i+length_lookback-1:i+length_lookback]
        data_x.append(features)
        data_y.append(targets)
    return np.array(data_x), np.array(data_y)


# Function for scaling the data. 
def data_scale(x_data=None, y_data=None, scaler_x=None, scaler_y=None, path=None):


    if scaler_x == None and scaler_y == None:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x_data_scaled, y_data_scaled = \
        scaler_x.fit_transform(x_data), scaler_y.fit_transform(y_data.values.reshape(-1, 1)).reshape(-1)
    elif scaler_x == None or scaler_y == None:
        raise ValueError("Both 'scaler_x' or 'scaler_y' should be either set or None.")
    elif scaler_x != None and scaler_y != None:
        x_data_scaled, y_data_scaled = \
        scaler_x.transform(x_data), scaler_y.transform(y_data.values.reshape(-1, 1)).reshape(-1)
    if path is not None:
        joblib.dump(scaler_x, f'{path}scaler_x.joblib')
        joblib.dump(scaler_y, f'{path}scaler_y.joblib')
    return x_data_scaled, y_data_scaled, scaler_x, scaler_y
# Data Split
def dataset_division(dataset=None, division_year=None, shuffling=False):
    # Select data from the start year up to the division year for training, reset the index, and append to the training DataFrame.
    data_train = dataset[dataset.datetime >= f'{division_year}-01-01'].reset_index(drop=True)
    
    # Select data from the division year onward for testing, reset the index, and append to the testing DataFrame.
    data_test = dataset[dataset.datetime < f'{division_year}-01-01'].reset_index(drop=True)

    if shuffling == True:

        data_train = shuffle(data_train, random_state=42)
    
        data_test = shuffle(data_test, random_state=42)
        
    return data_train, data_test


def data_prepare(data_train=None, data_test=None, path=None, length_lookback=None):

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(data_train.iloc[:, 6:-1])
    scaler_y.fit(data_train.iloc[:, -1].values.reshape(-1, 1))
    
    # Initialize dictionaries to store scaled test datasets.
    x_test_scaled = {}
    y_test_scaled = {}
    x_test = {}  # Missing declaration in your provided code.
    y_test = {}  # Missing declaration in your provided code.
    x_train_scaled = {}
    y_train_scaled = {}
    x_train = {}  # Missing declaration in your provided code.
    y_train = {}  # Missing declaration in your provided code.
        # Loop over each station name from the list of station IDs.
    for station_name in data_test.station_id.unique():

        # Extract and store the features for the test data for each station.
        x_train[station_name] = data_train[data_train.station_id == station_name].iloc[:, 6:-1]
        # Extract and store the target variable for the test data for each station.
        y_train[station_name] = data_train[data_train.station_id == station_name].iloc[:, -1]
    
        x_train_scaled[station_name], y_train_scaled[station_name], _, _ = data_scale(x_train[station_name], y_train[station_name], scaler_x=scaler_x, scaler_y=scaler_y)
        
        x_train_scaled[station_name], y_train_scaled[station_name] = add_lookback(x_train_scaled[station_name], y_train_scaled[station_name], length_lookback=length_lookback)
        # Extract and store the features for the test data for each station.
        x_test[station_name] = data_test[data_test.station_id == station_name].iloc[:, 6:-1]
        # Extract and store the target variable for the test data for each station.
        y_test[station_name] = data_test[data_test.station_id == station_name].iloc[:, -1]
    
        x_test_scaled[station_name], y_test_scaled[station_name], _, _ = data_scale(x_test[station_name], y_test[station_name], scaler_x=scaler_x, scaler_y=scaler_y)
        x_test_scaled[station_name], y_test_scaled[station_name] = add_lookback(x_test_scaled[station_name], y_test_scaled[station_name], length_lookback=length_lookback)
    
    return x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled, scaler_x, scaler_y, y_train, x_test, y_test



def data_split(dataset):
    
    cols_to_drop = ['min', 'max', 'mean', 'median']
    dataset = dataset.drop(columns=[col for col in cols_to_drop if col in dataset.columns])
  

    data_train, data_test = dataset_division(dataset, division_year=2007, shuffling=False)


    data_train, data_test = add_properties(data_train, data_test)

    return data_train, data_test
