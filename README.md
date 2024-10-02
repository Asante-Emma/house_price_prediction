# **House Price Prediction using Artificial Neural Networks**

## Content
- [Problem Statement](#problem-statement)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Description](#data-description)
- [Preprocessing](#preprocessing)
- [Model Building](#model-building)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Conclusion](#conclusion)

## **Problem Statement**

**Situation**

In the real estate market, accurate pricing is crucial for both buyers and sellers. Property prices depend on various factors, including the size, location, and amenities of the property. With the increasing availability of data, machine learning models have the potential to predict house prices more accurately than traditional methods, helping stakeholders make informed decisions.

**Complication**

However, predicting house prices is complex due to the variability in property characteristics and market conditions. Traditional pricing methods often fail to capture the nuances of the real estate market, leading to inaccurate estimates. Additionally, the presence of missing data, outliers, and inconsistent feature scales further complicates the model development process.

**Question**

How can we leverage machine learning techniques to build a robust model that accurately predicts house prices based on a wide range of property features? What preprocessing steps are necessary to handle data inconsistencies, and how can the model's performance be evaluated and validated effectively?

## **Overview**
This project involves predicting house prices using an Artificial Neural Network (ANN) model. The project follows a structured approach that includes data preprocessing, model building, evaluation, and deployment.

## **Project Structure**
- **data/**: Contains the raw and processed datasets.
- **models/**: Contains the saved model and scaler.
- **notebooks/**: Jupyter notebooks used for exploration, training, and evaluation.
- **src/**: Python scripts for data preprocessing, model training, and prediction.
- **deployment/**: Flask API for making predictions.
- **README.md**: Project documentation.

## **Data Description**
The dataset consists of various features related to house properties, such as area, number of rooms, and location. The target variable is the logarithm of the house price (`Price_log`).

### Key Features:
- **Carpet Area**: The area of the house in square feet.
- **Bathroom**: Number of bathrooms.
- **Balcony**: Number of balconies.
- **Super Area**: Total area including walls and other areas.
- **Current_Floor**: The floor on which the house is located.
- **Total_Floors**: Total number of floors in the building.
- **Location_freq**: Frequency encoding of the house location.
- **Society_freq**: Frequency encoding of the housing society.
- **Car_Parking_freq**: Frequency encoding of the car parking space.

## **Preprocessing**
The data preprocessing involved:
1. Dropping columns with more than 30% missing values.
2. Converting categorical variables to numeric.
3. Handling missing values by imputing them.
4. Removing outliers by clipping data to the 1st and 99th percentiles.
5. Scaling numerical features using `StandardScaler`.

## **Model Building**
An ANN model was built using the following architecture:
- **Input Layer:** 44 features matching the number of input features.
- **First Hidden Layer:** 64 neurons with ReLU activation.
- **Second Hidden Layer:** 128 neurons with ReLU activation.
- **Dropout Layer:** 0.3 dropout rate to prevent overfitting.
- **Third Hidden Layer:** 64 neurons with ReLU activation.
- **Dropout Layer:** 0.3 dropout rate to prevent overfitting.
- **Fourth Hidden Layer:** 32 neurons with ReLU activation.
- **Output Layer:** A single neuron with linear activation for predicting the log-transformed price.

The model was compiled using the Adam optimizer and Mean Squared Error loss. Early stopping was implemented with a patience of 10 epochs, restoring the best weights during training, and the model was trained for up to 100 epochs with a batch size of 32.

### Training Details:
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Epochs**: 100
- **Batch Size**: 32

## **Evaluation**
The model was evaluated using:
- **Mean Absolute Error (MAE)**: 0.1367
- **Mean Squared Error (MSE)**: 0.0585
- **R Squared (R²)**: 0.7560

## **Deployment**
A Flask API was created to serve the model for real-time predictions. The API accepts JSON input and returns the predicted price.

### API Endpoints:
- **/predict**: Takes house features as input and returns the predicted price.

### How to Use
1. **Clone the repository**:
    ```bash
    git clone https://github.com/Asante-Emma/house_price_prediction.git
    ```
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the API**:
    ```bash
    python app.py
    ```
4. **Make Predictions**:
    Send a POST request to `/predict` with the required features.

## **Conclusion**
The **House Price Prediction project** demonstrates the effective use of Artificial Neural Networks (ANNs) to forecast real estate prices based on various property features. By addressing challenges like missing data and outliers through robust preprocessing techniques, the model achieved a satisfactory performance, with an R² score of 0.7560. The implementation of a Flask API enables real-time predictions, making the model accessible for practical use in the real estate market. This project highlights the potential of machine learning in enhancing decision-making for buyers and sellers, paving the way for more accurate property valuations.
