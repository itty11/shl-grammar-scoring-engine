
# Tune Ridge regularization using the existing text + audio features.

exec(open("baseline_multimodal.py", encoding="utf-8").read().split(
    'model = Pipeline(['
)[0])

from sklearn.linear_model import Ridge

# The preprocessing pipeline and train/validation split were
# defined above by the existing baseline script.
X_train_features = preprocessor.fit_transform(X_train)
X_val_features = preprocessor.transform(X_val)

print("\nRIDGE ALPHA COMPARISON")
print("alpha | train RMSE | validation RMSE | Pearson")

results = []

for alpha in [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]:
    model = Ridge(alpha=alpha)
    model.fit(X_train_features, y_train)

    train_pred = np.clip(model.predict(X_train_features), 0, 5)
    val_pred = np.clip(model.predict(X_val_features), 0, 5)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_pred)
    )
    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_pred)
    )
    pearson = pearsonr(y_val, val_pred)[0]

    results.append({
        "alpha": alpha,
        "train_rmse": train_rmse,
        "validation_rmse": val_rmse,
        "pearson": pearson
    })

    print(
        f"{alpha:5.1f} | {train_rmse:10.4f} | "
        f"{val_rmse:15.4f} | {pearson:.4f}"
    )

results_df = pd.DataFrame(results)
results_df.to_csv(
    DATASET / "ridge_tuning_results.csv",
    index=False
)

print("\nSaved ridge_tuning_results.csv")
