# 🌾 Cropify – Crop Recommendation System

**Cropify** is an interactive web application that recommends the most suitable crop to grow based on soil nutrients, weather conditions, and environmental parameters. It leverages **Machine Learning (Random Forest)** to make predictions and provides an intuitive **Streamlit interface** for farmers, students, or enthusiasts.

---

## **Features**

- Automatically downloads the latest crop recommendation dataset from Kaggle.  
- Normalizes and preprocesses the dataset for training.  
- Trains a **Random Forest Classifier** if no pre-trained model exists.  
- Saves and loads the trained model and scaler for fast predictions.  
- User-friendly **Streamlit interface** with two-column input layout.  
- Default input values are set to **0**, allowing flexible user input.  
- Predicts the most suitable crop for the given farm parameters.  

---

## **Folder Structure**

Cropify/
│
├── data/ # Folder for CSV dataset
│ └── Crop_recommendation.csv
├── cropify_app.py # Main Streamlit application
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## **Installation & Setup**

1. **Clone the repository:**

```bash
git clone https://github.com/Rohith723/Cropify.git
cd Cropify

2. **Install dependencies:**

pip install -r requirements.txt

3. **Run the Streamlit app:**

streamlit run cropify_app.py
