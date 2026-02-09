# Fetal Movement Detection Project

A complete end-to-end machine learning and deep learning project for detecting fetal movement using accelerometer data from pregnant women.

## 📋 Overview

This project uses accelerometer signals recorded from the abdominal wall of pregnant women to detect fetal movements. The dataset contains recordings from 16 different pregnant women with 3-axis acceleration data sampled at 500Hz.

## 📁 Project Structure

```
islamabad/
├── data/
│   ├── *_signal.mat    # 3-axis accelerometer data (X, Y, Z)
│   ├── *_bp.mat        # Maternal perception markers (labels)
│   └── readme.txt      # Dataset description
├── Fetal_Movement_Detection.ipynb  # Main analysis notebook
└── README.md           # This file
```

## 🔬 Dataset Information

- **Source**: Accelerometer recordings from pregnant women
- **Subjects**: 16 pregnant women
- **Sensor**: ADXL355 accelerometer (ANALOG DEVICES)
- **Sampling Frequency**: 500 Hz
- **Data Format**: MATLAB .mat files
  - `*_signal.mat`: 3-axis acceleration data
  - `*_bp.mat`: Maternal perception markers (movement annotations)

## 🛠️ Requirements

The notebook includes an installation cell, but here are the main dependencies:

```
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn
imbalanced-learn
tensorflow
keras
```

## 🚀 How to Run

1. Open the `Fetal_Movement_Detection.ipynb` notebook in VS Code or Jupyter
2. Run the first cell to install all required libraries
3. Execute cells sequentially from top to bottom
4. Each cell produces immediate output (graphs, metrics, etc.)

## 📊 Notebook Contents

### 1. Library Installation & Imports
- All required packages installation
- Import statements for data science and ML libraries

### 2. Data Loading
- Load .mat files from data folder
- Display data structure and first rows
- Check and handle missing values

### 3. Data Balancing
- Visualize class distribution
- Apply SMOTE if dataset is imbalanced

### 4. Exploratory Data Analysis (EDA)
- Dataset overview and statistics
- Signal visualization (time-series plots)
- Distribution analysis (histograms)
- Correlation heatmap
- Box plots for outlier detection
- Skewness and kurtosis analysis
- Movement labels overlay on signals

### 5. Data Preprocessing
- StandardScaler normalization
- Sliding window segmentation (1-second windows, 50% overlap)
- Train/test split (80/20)

### 6. Machine Learning - Random Forest
- Train Random Forest classifier
- Confusion matrix visualization
- Classification report
- Metrics bar chart (Accuracy, Precision, Recall, F1)
- Feature importance plot

### 7. Deep Learning - BiLSTM
- Build Bidirectional LSTM architecture
- Training with early stopping
- Training vs Validation accuracy plot
- Training vs Validation loss plot
- Confusion matrix
- Classification report
- Metrics bar chart
- ROC curve with AUC score
- Precision-Recall curve

### 8. Model Comparison
- Side-by-side comparison of Random Forest vs BiLSTM
- Comparative metrics visualization

## 📈 Model Architecture

### Random Forest
- 100 estimators
- Max depth: 20
- Features: Flattened window samples

### BiLSTM
```
Input Layer (500 timesteps, 3 features)
    ↓
Bidirectional LSTM (64 units, return sequences)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (32 units)
    ↓
Dropout (0.3)
    ↓
Dense (32 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense (1 unit, Sigmoid)
```

## 📝 Notes

- All code is contained in a single Jupyter notebook
- Each analysis step is in a separate cell for easy execution
- Graphs and outputs are displayed inline
- The notebook is designed to run cell-by-cell in VS Code

## 👤 Author

Fetal Movement Detection Project

## 📄 License

This project is for educational and research purposes.
