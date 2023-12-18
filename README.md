# Training the model 

## What does this function app do ?

This function app is responsible for creating the AI models for each of the squares that will later be used to make predictions

## Step 1: Get the data for each square: list_folders()

The data for each square is stored in individual folders with the UUID of the squares. So naturally, our first step is to retrieve those folders

## Step 2: download the data: get_merged_df()

In each folder, the relevant data is stored in the "merged.csv" file. We retrieve that file and download it to a local pandas frame.

## Step 3: get the latest data: get_latest()

Return the latest relevant data for each the square.   

The issue we have been encountering is that there are many gaps in our data, if the gaps are too big, the data is unusable.   

To tackle this issue, we do the following:
- take the last 10 rows of the dataframe
- count the number of Nan values in each column and average them over all the columns
- if there are less than 75% Nan values, we repeat the process but look at 10 more rows
- if there are more than 75% Nan values, we return the rows we just analyzed as the latest data   

This helps us to get the relevant latest data without encountering too many Nan values

## Step 3bis: get the latest data (temporary): temp_get_latest()

Since we do not have the application constantly running, there is no recent data stored in the storage and the previous function would just return an empty dataframe.   

Instead, we just take the data from 8:50 to 17:20 on 7th December 2023 (a time interval where we have enough data for each square) and return that to the model.   

Keep in mind this is a TEMPORARY method that will be removed once all the part of the application are deployed

## Step 4: create models for each square: create_model_from_df()

Steps:

1. **Drop Timestamp Column:**
   - Removes the timestamp column from the DataFrame.

2. **Check for Empty Data:**
   - If the DataFrame has no columns or rows, the function returns `None` with a corresponding log message.

3. **Interpolate Missing Values:**
   - Uses linear interpolation to fill in missing values in the DataFrame.

4. **Data Preprocessing:**
   - Converts the DataFrame to a NumPy array and scales the data using the Standard Scaler.

5. **Create Sequences:**
   - Defines a window size and creates input-output sequences for the CNN model.

6. **Split Data:**
   - Divides the data into training and testing sets.

7. **Build CNN Model:**
   - Constructs a simple CNN model using Keras with Conv1D, MaxPooling1D, Flatten, and Dense layers.

8. **Compile Model:**
   - Compiles the model using the Adam optimizer and Mean Squared Error (MSE) loss.

9. **Define Early Stopping:**
   - Implements early stopping to prevent overfitting during training.

10. **Train the Model:**
    - Trains the CNN model on the training data with validation on the testing data.

11. **Evaluate Model:**
    - Evaluates the trained model on the testing set and prints the test loss.

12. **Return Model:**
    - Returns the trained CNN model.

## Step 5: delete and store

Finally, for each square, we delete the previously stored model and store the new one in the corresponding folder. The model is stored in a pickle file for easy access