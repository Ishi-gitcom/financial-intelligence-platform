from sklearn.preprocessing import StandardScaler
def prepare_train_test_data(df, features, target="Target", test_size=0.2):

    # Features and target
    X = df[features]
    y = df[target]

    # Time-based split
    split = int(len(X) * (1 - test_size))
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    # Scale features
    X_train_raw = X_train.copy()
    X_test_raw = X_test.copy()

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Training Target Range:")
    print(y_train.min(), y_train.max())

    print("Testing Target Range:")
    print(y_test.min(), y_test.max())
    return (
    X_train_raw,
    X_test_raw,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    scaler
)