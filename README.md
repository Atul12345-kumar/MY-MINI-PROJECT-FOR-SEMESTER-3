# 🚗 Used Car Price Lab — Neural Network vs. Analytical Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Only%20Dependency-013243?logo=numpy&logoColor=white)
![GUI](https://img.shields.io/badge/GUI-Tkinter%20(Built--in)-green)
![Status](https://img.shields.io/badge/Maintained-Yes-brightgreen)

An interactive machine learning laboratory built **entirely from scratch using pure NumPy** (no `scikit-learn`, `PyTorch`, `TensorFlow`, or `matplotlib` required). 

This project compares a **Multi-Layer Perceptron (MLP) Neural Network** trained via Mini-Batch Adam against a **Closed-Form Analytical Model (OLS / Ridge Regression)** for predicting used car prices in the Indian market (measured in **INR Lakh**). It includes both a multi-tab **Tkinter GUI** with custom canvas charting and a full-featured **CLI mode**.

---

## ✨ Key Features

* **100% From-Scratch ML Engine**:
  * **Analytical Model**: Ordinary Least Squares (OLS) and $L_2$-regularized Ridge Regression solved via the Normal Equation with optional 2nd-degree polynomial and pairwise interaction expansion.
  * **Neural Network (MLP)**: Configurable architecture (1 to 6 hidden layers), ReLU/Tanh activations, He/Xavier weight initialization, Mini-Batch **Adam Optimizer**, $L_2$ weight decay, and **Early Stopping**.
* **Zero Plotting Dependencies**: Custom-built vector plotting engine using native `tkinter.Canvas` for scatter plots, training/validation loss curves, grouped feature importance bar charts, and depreciation curves.
* **Synthetic Market Simulator & Custom CSV Support**: Built-in data generator modeling realistic vehicle depreciation, mileage wear, fuel type, transmission, and brand tier effects—or load your own real-world dataset.
* **Model Explainability**: Built-in **Permutation Feature Importance** (measuring RMSE degradation across shuffled repeats) and optional **5-Fold Cross-Validation**.
* **Interactive "What-If" Price Predictor**: Input custom vehicle specs to compare instant predictions from both models alongside a 20-year projected depreciation curve.
* **One-Click Artifact Export**: Exports the active dataset, test set predictions, formatted evaluation report, and hyperparameter configuration.

---

## 🛠️ Installation & Requirements

The project requires **Python 3.8+** and **`numpy`**. The graphical interface uses `tkinter`, which comes pre-installed with standard Python distributions on Windows and macOS.

```bash
# 1. Clone the repository
git clone [https://github.com/your-username/used-car-price-lab.git](https://github.com/your-username/used-car-price-lab.git)
cd used-car-price-lab

# 2. Install NumPy
pip install numpy
```

> **Linux Users**: If `tkinter` is not installed on your system, install it via `sudo apt install python3-tk` (Ubuntu/Debian), or the script will automatically fall back to the interactive text-mode CLI.

---

## 🚀 Usage

### 1. Launch the Interactive GUI (Default)
```bash
python used_car_price_lab.py
```

The GUI is organized into four workflow tabs:
1. **Data**: Configure sample size ($N$), price noise %, random seed, and test split %, or load/save a CSV dataset. Preview the top 200 rows directly in the table.
2. **Models & Training**: Tune Analytical ($\lambda$, polynomial degree) and Neural Network hyperparameters (hidden layer sizes, activation, learning rate, $L_2$ decay, epochs, batch size, early-stopping patience). Toggle 5-fold CV and monitor real-time epoch logs.
3. **Results**: Compare test metrics (**MAE, RMSE, $R^2$, MAPE%**), inspect 4 interactive diagnostic charts, and export all run artifacts.
4. **Predict Your Car**: Enter custom vehicle attributes to compare point predictions and visualize a 0–20 year **What-If Depreciation Curve**.

### 2. Run in Command-Line Interface (CLI) Mode
For headless environments, servers, or quick terminal runs:
```bash
python used_car_price_lab.py --cli
```
Press `Enter` at each prompt to accept the default hyperparameters, view the full evaluation table in your terminal, and interactively query car prices.

---

## 🧠 Model Architecture & Mathematical Formulation

### 1. Analytical Model (`AnalyticalModel`)
Solves for parameters $\theta$ in closed form using the Moore-Penrose pseudoinverse (`np.linalg.pinv`) after $z$-score feature standardization:

$$\theta = (A^T A + \lambda P)^{-1} A^T y$$

* $A = [\mathbf{1}, Z]$ is the design matrix with an unpenalized intercept column ($P_{0,0} = 0$).
* When `poly=2`, the 9 base features are expanded to include all quadratic squares ($x_i^2$) and pairwise interaction terms ($x_i x_j$), expanding the feature space from **9 to 54 features**.

### 2. Neural Network (`NeuralNetwork`)
A fully connected feedforward network trained on standardized inputs and targets:
* **Weight Initialization**: He scaling ($\sqrt{2/n_{\text{in}}}$) for `relu`; Xavier scaling ($\sqrt{1/n_{\text{in}}}$) for `tanh`.
* **Optimization**: Mini-batch **Adam** ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$) minimizing Mean Squared Error (MSE) with $L_2$ weight regularization.
* **Validation & Early Stopping**: Automatically reserves a 10% internal validation split (`Xv`, `yv`), tracks best weights across epochs, and halts training if validation loss fails to improve by at least $10^{-6}$ for `patience` consecutive epochs.

---

## 📊 Dataset Schema & Features

### Built-In Simulation Rules
When using simulated data, prices (in **INR Lakh**) are generated using realistic domain rules:
* **Base price** scales with `engine_cc`, `brand_tier` (*Economy: 1.0×, Mid: 1.45×, Premium: 2.4×*), `transmission` (*Automatic: 1.18×*), and `fuel` (*Petrol: 1.0×, Diesel: 1.10×, CNG: 0.92×*).
* **Exponential age depreciation** ($\approx e^{-0.095 \times \text{age}}$), **mileage wear penalty**, **ownership discount** ($6\%$ per additional owner), and configurable Gaussian noise.

### Loading a Custom CSV
To load your own dataset via the GUI, your CSV must contain **at least 50 rows** and the following **8 required columns**:

| Column Name | Type | Allowed Values / Range | Description |
| :--- | :--- | :--- | :--- |
| `age_years` | `int` | `1` – `40` | Age of the vehicle in years |
| `km_driven` | `int` | `0` – `1,000,000` | Total odometer reading in kilometers |
| `engine_cc` | `int` | `500` – `6000` | Engine displacement in cubic centimeters |
| `fuel` | `str` | `Petrol`, `Diesel`, `CNG` | Fuel type (case-insensitive) |
| `transmission` | `str` | `Manual`, `Automatic` | Transmission type |
| `owners` | `int` | `1`, `2`, `3` | Number of previous owners |
| `brand_tier` | `str` | `Economy`, `Mid`, `Premium` | Market segment of the manufacturer |
| `price_lakh_inr`| `float` | `> 0.0` | Target resale price in INR Lakhs |

One-hot encoding maps these 7 inputs into **9 numerical features** (`FEATS`):
`age_years`, `km_driven`, `engine_cc`, `owners`, `fuel_Diesel`, `fuel_CNG`, `trans_Automatic`, `brand_Mid`, `brand_Premium`.

---

## ⚙️ Hyperparameter Reference

| Parameter | Key | Default | Valid Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset Size** | `n` | `1500` | `100` – `20000` | Number of synthetic vehicle records |
| **Noise %** | `noise` | `5.0` | `0.0` – `50.0` | Multiplicative random Gaussian noise (%) |
| **Random Seed** | `seed` | `42` | Any `int` | Seed for reproducible splits and weights |
| **Test Split %** | `test` | `20` | `5` – `50` | Percentage of data held out for testing |
| **Ridge Lambda** | `lam` | `0.0` | `>= 0.0` | $L_2$ penalty for Analytical model (`0` = OLS) |
| **Polynomial Degree**| `poly` | `1` | `1` or `2` | `1` = linear terms; `2` = squares & interactions |
| **Hidden Layers** | `hidden` | `64,32` | 1–6 layers (`1`–`512`) | Comma-separated neuron counts per layer |
| **Activation** | `act` | `relu` | `relu`, `tanh` | Hidden layer activation function |
| **Learning Rate** | `lr` | `0.01` | `> 0.0` | Step size for Adam optimizer |
| **L2 Weight Decay** | `l2` | `0.0001` | `>= 0.0` | Weight regularization strength for MLP |
| **Max Epochs** | `epochs` | `400` | `1` – `3000` | Maximum training passes over the dataset |
| **Batch Size** | `batch` | `64` | `8` – `2048` | Mini-batch size for stochastic gradient descent |
| **Patience** | `patience` | `25` | `1` – `500` | Early stopping patience (epochs without improvement) |
| **5-Fold CV** | `cv` | `False` | `True` / `False` | Computes 5-fold cross-validated $R^2$ (mean $\pm$ std) |

---

## 📂 Output & Exported Artifacts

Clicking **Export all...** in the GUI (or running `--cli`) automatically creates a `car_project_output/` directory containing:

```text
car_project_output/
├── dataset.csv       # Full dataset used during the experiment
├── predictions.csv   # Test set Actual vs. Analytical vs. NeuralNet predictions
├── report.txt        # Formatted summary of metrics, 5-fold CV, and feature importances
└── settings.json     # Exact hyperparameter configuration for reproducibility
```

### Sample Console / Report Output
```text
USED CAR PRICE LAB - REPORT
(dataset may be simulated; prices in INR lakh)

Rows: 1500   test split: 20%   seed: 42
Analytical: lambda=0.0 poly=1   NN: hidden=(64, 32) act=relu lr=0.01 l2=0.0001

Model                               MAE     RMSE       R2    MAPE%
Analytical (Ridge/OLS)           0.4128   0.5842   0.8914    14.23
Neural Network (MLP)             0.1984   0.2891   0.9734     6.18
```

---

## 👤 Author

**Atul Kumar**  
*MBA — Data Science & Artificial Intelligence*  

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
