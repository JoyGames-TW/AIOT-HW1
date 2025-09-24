import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Linear Regression with Custom Parameters",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

col1, col2 = st.columns([0.2, 0.8])

with col1:
    st.header("Parameters")
    n = st.slider("Number of samples (n)", 100, 5000, 1000)
    a = st.slider("Slope (a)", -10.0, 10.0, 1.0)
    var = st.slider("Noise variance (var)", 0.0, 1000.0, 1.0)

with col2:
    # Generate synthetic data
    np.random.seed(42)
    x = np.random.randn(n)
    noise = np.random.normal(0, np.sqrt(var), n)
    y = a * x + noise

    # Create DataFrame
    data = pd.DataFrame({'x': x, 'y': y})

    # Split data
    X = data[['x']]
    y_data = data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}

    # Display results
    st.header("Model Results")
    for name, metrics in results.items():
        st.write(f"**{name}**: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

    # Plot
    st.header("Data and Prediction")
    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, label="Actual")
    # Predict on test set
    best_model = LinearRegression()
    best_model.fit(X_train_scaled, y_train)
    y_pred_plot = best_model.predict(X_test_scaled)
    ax.plot(X_test, y_pred_plot, color='red', label="Predicted")
    ax.legend()

    # Calculate residuals and find top 5 worst points
    residuals = np.abs(y_test - y_pred_plot)
    worst_indices = np.argsort(residuals)[-5:]  # Top 5 largest residuals

    # Label the worst points
    for i in worst_indices:
        ax.annotate(f'Worst {list(worst_indices).index(i)+1}', (X_test.iloc[i], y_test.iloc[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8),
                    color='blue')

    st.pyplot(fig)