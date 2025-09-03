🌾 Cropify – Crop Recommendation System

Cropify is an interactive web application that recommends the most suitable crop to grow based on soil nutrients, weather conditions, and environmental parameters. It leverages Machine Learning (Random Forest) to make predictions and provides an intuitive Streamlit interface for farmers, students, or enthusiasts.

Features

Automatically downloads the latest crop recommendation dataset from Kaggle.

Normalizes and preprocesses the dataset for training.

Trains a Random Forest Classifier if no pre-trained model exists.

Saves and loads the trained model and scaler for fast predictions.

User-friendly Streamlit interface with two-column input layout.

Default input values are set to 0, allowing flexible user input.

Predicts the most suitable crop for the given farm parameters.

Folder Structure
Cropify/
│
├── data/                  # Folder for CSV dataset
│   └── Crop_recommendation.csv
├── cropify_app.py         # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

Installation & Setup

Clone the repository:

git clone https://github.com/Rohith723/Cropify.git
cd Cropify


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run cropify_app.py

Usage

Open the app in your browser (Streamlit will provide a local URL).

Enter your farm parameters:

Nitrogen (N)

Phosphorous (P)

Potassium (K)

Temperature (°C)

Humidity (%)

Soil pH

Rainfall (mm)

Click Predict Crop to get the recommended crop.

The app will show the recommended crop based on your inputs.

Dependencies

Python 3.x

Streamlit

Pandas

Scikit-learn

NumPy

KaggleHub

Install all dependencies using:

pip install -r requirements.txt

Notes

The first run will automatically download the latest dataset from Kaggle.

The model will train automatically if no saved model exists, otherwise it will load the pre-trained pickle files for faster predictions.

Default input values are set to 0 for flexibility.

License

This project is open-source and available under the MIT License.
