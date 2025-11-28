# Engine-Torque-Nitrous-Oxide-Prediction-Deep-Neural-Network

Engine Torque &amp; Nitrous Oxide Prediction — Deep Neural Network

This project builds a Deep Neural Network (DNN) to predict **engine torque** and **nitrous oxide emissions** from engine operating parameters. The model learns hidden nonlinear relationships between fuel rate, engine speed, and output performance metrics.

The dataset provided contains raw experimental data in transposed Excel format.  
The pipeline automatically handles transpose, cleaning, coercing to numeric types, scaling, training, and evaluation.

---

##  Project Overview

This repository demonstrates a complete **multi-output regression** workflow, including:

- Clean ingestion of transposed Excel data
- Automatic data restructuring and type correction
- Feature scaling using StandardScaler
- Multi-output Deep Neural Network (2 regression targets)
- Training, evaluation, visualization, and inference
- Export of the trained model and scaler

The model predicts:
- **Engine Torque (Nm)**
- **Nitrous Oxide Emission Levels (ppm)**

---

##  Repository Structure
engine-prediction-dnn/
│── README.md
│── model.py # Training pipeline
│── predict.py # Inference script
│── keras_colab.py # Colab-ready training script (GPU supported)
│── data/
│ ├── inputs_Engine.xlsx
│ └── output_Engine.xlsx
│── outputs/ # Confusion matrix, regression plots, metrics
│── saved_model/ # Trained model (.h5 / .joblib), scaler, features
│── requirements.txt


---

##  Model Architecture

The default DNN uses:

- Dense(20, relu)
- Dense(32, relu)
- Dense(2, linear) → Predicts Torque & NOₓ simultaneously

**Loss:** Mean Squared Error (MSE)  
**Optimizer:** SGD or Adam  
**Metrics:** MSE, MAE, R² (via sklearn)

---

## How It Works

### Data Loading & Preprocessing
- Excel inputs are loaded from `inputs_Engine.xlsx` and `output_Engine.xlsx`.
- Files are **transposed automatically** to convert them into row-major format.
- Columns are renamed:
  - Fuel_Rate  
  - Speed  
  - Torque  
  - Nitrous_oxide_emissions
- Non-numeric data are coerced into numeric format.
- Missing rows are removed.
- Inputs and outputs are standardized using **StandardScaler**.

## Model Training
Run the pipeline:
python model.py --epochs 300 --batch 16

## Evaluation
Generated outputs include:

✔ Regression performance metrics
✔ Actual vs Predicted scatter plots
✔ Error distribution plots
✔ Saved model (.h5 or .joblib)
✔ Saved scaler (scaler.joblib)

All saved in:
outputs/
saved_model/

##  Results (Example)

MSE: Low (good fit)

Predicted Torque closely matches actual

Nitrous Oxide emission regression stable

Learned feature relationships consistent with expected engine behavior

##  Prediction on New Engine Data
Use:
python predict.py --model saved_model/engine_dnn.h5 --scaler saved_model/scaler.joblib --input new_engine_data.csv
Outputs saved to:
outputs/predictions.csv

##  Google Colab Ready

The file keras_colab.py automates:

TensorFlow installation

Excel upload

Training on GPU

Plotting results

Saving the .h5 model

Perfect for fast experimentation and cloud-based computation.

##  Requirements
pandas
numpy
tensorflow
scikit-learn
matplotlib
seaborn
openpyxl
joblib

## Install:
pip install -r requirements.txt

##  Summary

This project demonstrates a clean, industry-ready workflow for multi-output regression using deep learning. The model captures nonlinear relationships between engine parameters and their performance outputs, producing accurate predictions suitable for optimization, simulation, and control applications.


