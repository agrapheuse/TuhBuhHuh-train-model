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

    # Connect to our blob storage
    connection_string = "DefaultEndpointsProtocol=https;AccountName=datalaketuhbehhuh;AccountKey=C2te9RgBRHhIH8u3tydAsn9wNd4umdD2axq1ZdcfKh7CZRpL04+D4H6QinE/gckMTUA/dFj1kFpd+ASt4+/8ZA==;EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get all the folders in the csv container
    folders = list_folders(blob_service_client, "csv")

    # loop through all the folders (all the squares in our grid)
    for f in folders:
        # Get the merged.csv file for the folder (contains all the data for the square)
        merged_df = get_merged_df(blob_service_client, f)

        """ 
            It is supposed to take the latest data but that wouldnt be enough so we have a temporary
            function that takes the data from 8:50 to 17:20 on 7th December 2023 (a time interval where we
            have enough data for each square)

            As soon as we have the function apps and the projects deployed and running continuously, we can
            change this function to take the latest data.
        """
        # latest_data_df = get_latest(merged_df)
        latest_data_df = temp_get_latest(merged_df)

        # If there is no recent data, skip the square
        if latest_data_df is None:
            logging.info(f"Folder {f} has no recent data.")
            continue
        else:
            print(latest_data_df.describe())

        # Create the model unique to the square
        model = create_model_from_df(latest_data_df)

        # If there was an error in creating the model, skip the square
        if model is None:
            logging.info(f"No model created for folder {f}.")
            continue

        url_parts = f.split("/")

        # Find the index of "history" and remove it
        if "history" in url_parts:
            url_parts.remove("history")
        result_url = "/".join(url_parts)

        # Delete the previous model and store the new one
        delete_previous_model(blob_service_client, result_url)
        store_model(blob_service_client, model, result_url)
        logging.info(f"Model for folder {result_url} trained and stored.")

def list_folders(blob_service_client, container_name):
    # Connect to the container and extract a list of all the blobs
    container_client = blob_service_client.get_container_client(container_name)
    blob_list = container_client.list_blobs(name_starts_with="history/")

    folder_set = set()

    # Extract the folder names from the blob names and add them to a set of all the folder names
    for blob in blob_list:
        blob_name = blob.name
        folder_name = "/".join(blob_name.split('/')[:-1])
        folder_set.add(folder_name)

    return folder_set

# Download the merged.csv file for the square into a DataFrame
def get_merged_df(blob_service_client, folder):
    df = download_blob_to_file(blob_service_client, "csv", f"{folder}/merged.csv")
    return df

def get_latest(df):
    # Drop columns with no data
    columns_to_drop = []
    for i in df.columns:
        if df[i].count() == 0:
            columns_to_drop.append(i)

    df = df.drop(columns=columns_to_drop)

    # goes through the tail of the datafrale 10 rows at a time and checks if the average
    # number of non-null values is less than 75% of the number of rows. if it, that
    # means that there is not enough data for that square if we take more data so we return the dataframe
    df_without_timestamp = df.drop(columns='timestamp')
    for i in range(10, df.shape[0], 10):
        temp_df = df_without_timestamp.iloc[-i:]
        count_avg = temp_df.count().mean()
        if count_avg < i * 0.75:
            return temp_df

def temp_get_latest(df):
    # Takes the data from 8:50 to 17:20 on 7th December 2023 (a time interval where we
    # have enough data for each square)
    start_timestamp = '2023-12-07 08:50:00'
    end_timestamp = '2023-12-07 17:20:00'
    filtered_df = df[(df['timestamp'] >= start_timestamp) & (df['timestamp'] <= end_timestamp)]

    threshold = 0.5  # Set the threshold percentage
    nan_percentage = filtered_df.isna().mean()
    columns_with_high_nan = nan_percentage[nan_percentage > threshold].index.tolist()
    # Drop the column if it has more than 50% NaN values
    filtered_df = filtered_df.drop(columns=columns_with_high_nan)
    return filtered_df

def download_blob_to_file(blob_service_client, container_name, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_data = blob_client.download_blob()

    csv_file = StringIO(blob_data.readall().decode('utf-8'))
    df = pd.read_csv(csv_file)
    return df

def create_model_from_df(df):
    # Drop the timestamp column
    df = df.drop(columns='timestamp')

    # If there are no columns or rows, return None
    if df.shape[1] == 0 or df.shape[0] == 0:
        logging.info(f"Folder has no data.")
        return None

    # Interpolate the missing values
    df_interpolated_linear = df.interpolate(method='linear')
    
    # Convert the DataFrame to a NumPy array
    data_array = df_interpolated_linear.values

    # Normalize the data
    scaler = StandardScaler()
    data_array_scaled = scaler.fit_transform(data_array)

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

def delete_previous_model(blob_service_client, folder):
    # Delete the previous model in the specified folder
    blob_client = blob_service_client.get_blob_client(container="model", blob=f"{folder}/model.pkl")
    blob_client.delete_blob()
    logging.info('Previous model deleted.')

def store_model(blob_service_client, model, folder):
    # Store the new model in the specified folder as a pickle file
    model_pickle_string = pickle.dumps(model)

    blob_client = blob_service_client.get_blob_client(container="model", blob=f"{folder}/model.pkl")
    blob_client.upload_blob(model_pickle_string, blob_type="BlockBlob")

    logging.info('Model stored in blob storage.')