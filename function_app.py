import logging
import azure.functions as func
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from io import StringIO
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.callbacks import EarlyStopping
import numpy as np
import pickle


app = func.FunctionApp()

@app.schedule(schedule="0 0 * * 0", arg_name="myTimer", run_on_startup=True,
              use_monitor=False) 
def trainModel(myTimer: func.TimerRequest) -> None:
    logging.info('Python timer trigger function executed.')
    connection_string = "DefaultEndpointsProtocol=https;AccountName=datalaketuhbehhuh;AccountKey=C2te9RgBRHhIH8u3tydAsn9wNd4umdD2axq1ZdcfKh7CZRpL04+D4H6QinE/gckMTUA/dFj1kFpd+ASt4+/8ZA==;EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    folders = list_folders(blob_service_client, "csv")
    merged = create_model(blob_service_client, list(folders)[0])
    merged_on_time_df = merge_on_time(merged)
    model = create_model_from_df(merged_on_time_df)
    delete_previous_model()
    store_model(model)

def list_folders(blob_service_client, container_name):
    container_client = blob_service_client.get_container_client(container_name)
    blob_list = container_client.list_blobs(name_starts_with="history/")

    folder_set = set()

    for blob in blob_list:
        blob_name = blob.name
        # Extract folder name
        folder_name = "/".join(blob_name.split('/')[:-1])
        folder_set.add(folder_name)

    return folder_set

def create_model(blob_service_client, folder):
    container_client = blob_service_client.get_container_client("csv")
    blob_list = container_client.list_blobs(name_starts_with=folder)
    merged_df = pd.DataFrame(columns=["squareUUID", "timestamp", "BIKE", "CAR", "HEAVY", "HUMIDITY", "PEDESTRIAN", "PM10", "PM25", "TEMPERATURE"])
    for blob in blob_list:
        df = download_blob_to_file(blob_service_client, "csv", blob.name)
        merged_df = merge_two_df(merged_df, df)
    merged_df = merged_df.drop(columns="squareUUID", axis=1)
    merged_df.to_csv("merged.csv", index=False)
    return merged_df

def merge_two_df(df1, df2):
    appended_df = pd.concat([df1, df2], ignore_index=True)
    return appended_df

def download_blob_to_file(blob_service_client: BlobServiceClient, container_name, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_data = blob_client.download_blob()

    csv_file = StringIO(blob_data.readall().decode('utf-8'))
    df = pd.read_csv(csv_file)
    return df

def merge_on_time(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Resample the data to 5-minute intervals and calculate the mean
    resampled_data = df.resample('5T').mean()

    # Reset the index to make 'timestamp' a regular column again
    resampled_data.reset_index(inplace=True)
    for col in resampled_data.columns[1:]:
        resampled_data[col] = resampled_data[col].round(1)

    # Print the result
    resampled_data.to_csv("resampled_data.csv", index=False)
    return resampled_data

def create_model_from_df(df):
    start_timestamp = '2023-12-07 08:50:00'
    end_timestamp = '2023-12-07 17:20:00'
    filtered_df = df[(df['timestamp'] >= start_timestamp) & (df['timestamp'] <= end_timestamp)]
    threshold = 0.5  # Set the threshold percentage
    nan_percentage = filtered_df.isna().mean()
    columns_with_high_nan = nan_percentage[nan_percentage > threshold].index.tolist()
    filtered_df = filtered_df.drop(columns=columns_with_high_nan)

    for col in filtered_df.columns[1:]:
        filtered_df[col] = filtered_df[col].interpolate(method='linear', limit_direction='both')

    filtered_df = filtered_df.drop(['timestamp'], axis=1)
    filtered_df.to_csv("filtered_df.csv", index=False)

    # Convert the DataFrame to a NumPy array
    data_array = filtered_df.values

    # Normalize the data
    scaler = StandardScaler()
    data_array_scaled = scaler.fit_transform(data_array)
    print(data_array_scaled.shape)

    # Define window size and create sequences
    window_size = 36  # You can adjust this based on your data
    X, y = [], []

    for i in range(len(data_array_scaled) - window_size):
        X.append(data_array_scaled[i:i + window_size])
        y.append(data_array_scaled[i + window_size])

    X, y = np.array(X), np.array(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Build the CNN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(window_size, data_array.shape[1])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(data_array.shape[1], activation='linear'))

    model.compile(optimizer='adam', loss='mse')

    # Define early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'Model Test Loss: {loss}')
    return model

def delete_previous_model():
    connection_string = "DefaultEndpointsProtocol=https;AccountName=datalaketuhbehhuh;AccountKey=C2te9RgBRHhIH8u3tydAsn9wNd4umdD2axq1ZdcfKh7CZRpL04+D4H6QinE/gckMTUA/dFj1kFpd+ASt4+/8ZA==;EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container="model", blob=f"model.pkl")
    blob_client.delete_blob()
    logging.info('Previous model deleted.')

def store_model(model):
    model_pickle_string = pickle.dumps(model)

    connection_string = "DefaultEndpointsProtocol=https;AccountName=datalaketuhbehhuh;AccountKey=C2te9RgBRHhIH8u3tydAsn9wNd4umdD2axq1ZdcfKh7CZRpL04+D4H6QinE/gckMTUA/dFj1kFpd+ASt4+/8ZA==;EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container="model", blob=f"model.pkl")
    blob_client.upload_blob(model_pickle_string, blob_type="BlockBlob")

    logging.info('Model stored in blob storage.')