import pandas as pd


df = pd.read_csv("data/feature_engineered_music_data.csv")

print("DATASET LOADED SUCCESSFULLY!")

print("\nDATASET SHAPE:")
print(df.shape)



feature_columns = [
    "Duration_sec",
    "Artist_Count",
    "Language_English",
    "Language_Hindi",
    "Language_Malayalam",
    "Language_Tamil"]

X = df[feature_columns]

y = df["Popularity"]


print("\nINPUT FEATURES (X):")
print(X.head())

print("\nTARGET VARIABLE (y):")
print(y.head(10))

print("\nX SHAPE:")
print(X.shape)

print("\ny SHAPE:")
print(y.shape)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42)

print("\nTRAINING AND TESTING DATA:")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)




from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

print("\nLINEAR REGRESSION MODEL TRAINED SUCCESSFULLY!")



y_pred = linear_model.predict(X_test)

print("\nSAMPLE PREDICTIONS:")

results = pd.DataFrame({
    "Actual_Popularity": y_test.values,
    "Predicted_Popularity": y_pred})

print(results.head(10))




from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nLINEAR REGRESSION MODEL EVALUATION:")

print("Mean Absolute Error (MAE):", round(mae, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R² Score:", round(r2, 4))




from sklearn.ensemble import RandomForestRegressor

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42)

random_forest_model.fit(X_train, y_train)

print("\nRANDOM FOREST MODEL TRAINED SUCCESSFULLY!")



rf_pred = random_forest_model.predict(X_test)



rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRANDOM FOREST MODEL EVALUATION:")

print("Mean Absolute Error (MAE):", round(rf_mae, 2))
print("Root Mean Squared Error (RMSE):", round(rf_rmse, 2))
print("R² Score:", round(rf_r2, 4))



print("\nMODEL COMPARISON:")

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"],
    "MAE": [
        mae,
        rf_mae],
    "RMSE": [
        rmse,
        rf_rmse],
    "R2_Score": [
        r2,
        rf_r2]})

print(comparison.round(4))




import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7)

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--")

plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Actual vs Predicted Popularity - Linear Regression")

plt.xlim(0, 100)
plt.ylim(0, 100)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()