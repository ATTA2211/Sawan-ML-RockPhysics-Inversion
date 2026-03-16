import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def main():
    print("Quick test started...")

    X = pd.DataFrame({
        "DT4P": [70.1, 72.5, 68.9, 75.0, 71.2, 69.8],
        "LLD":  [12.0, 15.5, 13.2, 18.1, 14.0, 11.8],
        "NPHI": [0.21, 0.19, 0.24, 0.17, 0.20, 0.23],
        "PEF":  [3.1, 3.3, 3.0, 3.5, 3.2, 3.1],
        "RHOB": [2.32, 2.28, 2.35, 2.25, 2.30, 2.34],
        "GR":   [45, 52, 48, 60, 50, 47]
    })

    y = np.array([120.5, 118.2, 122.1, 116.8, 119.4, 121.0])

    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print("Quick test completed successfully.")
    print("Predictions:", np.round(y_pred, 2))
    print("RMSE:", round(rmse, 4))

if __name__ == "__main__":
    main()
