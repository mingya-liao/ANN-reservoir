# Reservoir Release Prediction with Artificial Neural Network (ANN)
This repository contains a PyTorch-based implementation of an Artificial Neural Network (ANN) model for predicting reservoir release (Qt) using hydrological and meteorological features such as reservoir storage (St), net inflow (It), and day of year (Dt). The model is designed with modular code architecture, enabling easy customization and extension for hydrological forecasting tasks.

## Key Features
- **Data Preprocessing Pipeline**: Handles raw reservoir operational data (normalization, feature engineering, train/validation/test splitting) with configurable time-shift features for inflow and release variables.
- **Modular ANN Architecture**: A flexible linear model with configurable hidden layers, dropout regularization, and Xavier/uniform parameter initialization for stable training.
- **Advanced Training Utilities**: Implements polynomial decay learning rate scheduling with warm-up, Adam optimizer with weight decay, and early stopping based on Nash-Sutcliffe Efficiency (NSE) on the validation set.
- **Comprehensive Evaluation**: Computes NSE (a key hydrological metric) for train/validation/test sets and generates visualizations of predicted vs. observed reservoir release values.
- **Reproducibility**: Fixed random seeds and configurable experimental parameters ensure consistent and repeatable results.

## Workflow Overview
1. **Data Preparation**: `prepare_dataset.py` reads raw reservoir data, normalizes features (e.g., DOY), creates time-shifted inflow/release features (It±n, Qt-n), and splits data into train/validation/test sets.
2. **Model Definition**: `model.py` implements a base neural network class with optimizer/lr scheduler configuration and a linear ANN model with ReLU activation and dropout.
3. **Training Pipeline**: `train_utils.py` manages the training loop (batch training, validation, early stopping) and `main.py` orchestrates the end-to-end experiment (parameter parsing, model initialization, training, and evaluation).
4. **Visualization**: `plot_utils.py` generates comparative plots of predicted vs. observed release values across training/validation/test periods to assess model performance.

## Dependencies
- Python 3.7+
- PyTorch >= 1.9.0
- NumPy, Pandas, Matplotlib, Scipy
- TensorBoardX (for training metrics visualization)

## Usage
1. Configure experimental parameters (data path, model hyperparameters, training settings) via command-line arguments or modify the default values in `main.py`.
2. Run the training pipeline:
   ```bash
   python main.py --data_path ../data/records/ --reservoir_ID 41 --model_name ANN --hidden_channels 8 --tot_updates 2000
   ```
3. Model checkpoints and evaluation plots (training/validation/test comparison) are saved to the `./result/` directory.

## Evaluation Metric
The model uses Nash-Sutcliffe Efficiency (NSE) (1 - NSE_loss) as the primary evaluation metric, where values closer to 1 indicate better predictive performance (a standard metric in hydrological modeling).

## License
This project is open-source and available under the MIT License. Feel free to use, modify, and distribute the code for academic or research purposes.

