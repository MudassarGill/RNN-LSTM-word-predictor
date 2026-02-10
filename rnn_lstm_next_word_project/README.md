# RNN/LSTM Next Word Prediction Project

This project implements next word prediction using three different Recurrent Neural Network (RNN) architectures: **Simple RNN**, **Deep RNN**, and **LSTM**.

The project is structured to run entirely within Jupyter Notebooks (`.ipynb`), making it easy to visualize data, train models, and generate text interactively.

## 📂 Project Structure

```
rnn_lstm_next_word_project/
├── data/
│   ├── dataset.txt          # The raw text data for training
│   ├── X.npy                # Processed feature sequences
│   ├── y.npy                # Processed target labels
│   ├── tokenizer.pkl        # Tokenizer object for converting text/indices
│   ├── simple_rnn_model.keras  # Trained Simple RNN model
│   ├── deep_rnn_model.keras    # Trained Deep RNN model
│   └── lstm_model.keras        # Trained LSTM model
│
├── src/
│   ├── prepare_data.ipynb      # Step 1: Data Preprocessing
│   ├── train_simple_rnn.ipynb  # Step 2a: Train Simple RNN
│   ├── train_deep_rnn.ipynb    # Step 2b: Train Deep RNN
│   ├── train_lstm.ipynb        # Step 2c: Train LSTM
│   └── predict.ipynb           # Step 3: Text Generation/Prediction
│
└── README.md
```

## 🚀 Usage Instructions

### 1. Data Preparation
- Place your text data in `data/dataset.txt`.
- Open `src/prepare_data.ipynb` and run all cells.
- This will tokenize the text, create sequences, and save `X.npy`, `y.npy`, and `tokenizer.pkl` to the `data/` folder.

### 2. Model Training
You can choose to train any of the following models:

#### **A. Simple RNN**
- Open `src/train_simple_rnn.ipynb`.
- Run all cells to train a basic RNN model.
- The model will be saved as `data/simple_rnn_model.keras`.

#### **B. Deep RNN**
- Open `src/train_deep_rnn.ipynb`.
- Run all cells to train a deep RNN with multiple layers and Dropout.
- The model will be saved as `data/deep_rnn_model.keras`.

#### **C. LSTM (Long Short-Term Memory)**
- Open `src/train_lstm.ipynb`.
- Run all cells to train an LSTM model, which generally captures longer dependencies better than simple RNNs.
- The model will be saved as `data/lstm_model.keras`.

### 3. Prediction / Text Generation
- Open `src/predict.ipynb`.
- You can select which model to load by uncommenting the appropriate line (default is `simple_rnn_model.keras`).
- Run the notebook to generate text based on a seed sentence.

## 🧠 Model Architectures

### Simple RNN
- **Embedding Layer**: Converts word indices to dense vectors.
- **SimpleRNN Layer(s)**: Basic recurrent layers.
- **Dense Output**: Predicts the probability of the next word across the vocabulary.

### Deep RNN
- **Embedding Layer**
- **Multiple SimpleRNN Layers**: Stacked layers with `return_sequences=True` for intermediate layers.
- **Dropout**: Added to prevent overfitting.
- **Dense Output**

### LSTM
- **Embedding Layer**
- **LSTM Layer**: More complex recurrent unit capable of learning long-term dependencies. `return_sequences=False` (Many-to-One architecture).
- **Dropout**
- **Dense Output**

## 📦 Requirements
- TensorFlow / Keras
- NumPy
- Matplotlib
- Pickle (standard library)

Ensure you have the necessary libraries installed:
```bash
pip install tensorflow numpy matplotlib
```
