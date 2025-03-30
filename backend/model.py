import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================
# 1️⃣ Load and Preprocess Data
# ==============================
def load_and_preprocess_data(file_path):
    """
    Load CSV and preprocess it for anomaly detection.
    - Uses "hour" and "region" as main features.
    - Infers "is_present" without using duration.
    """
    df = pd.read_csv(file_path)

    # ✅ Parse 'date' column
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")

    # ✅ Create "day_index" to encode the date as a number
    df["day_index"] = (df["date"] - df["date"].min()).dt.days

    # ✅ Drop 'duration' since it's not relevant for anomaly detection
    features = ["day_index", "hour", "region"]
    X = df[features].values  # Only using day_index, hour, and region

    # ✅ Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, scaler



# ==============================
# 2️⃣ Define LSTM Forecasting Model
# ==============================
class LSTMForecastingModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout_rate=0.2):
        """
        LSTM model to learn normal movement patterns (hour & region).
        input_size=2 because we only use (hour, region) as features.
        """
        super(LSTMForecastingModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Encoder
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Dropout before the output layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Fully connected output layer (outputs anomaly score)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Output anomaly score
        return output



# ==============================
# 3️⃣ Train the Model
# ==============================
def train_model(train_data, model, num_epochs=50, learning_rate=0.001):
    """
    Train the LSTM forecasting model.
    """

    # Convert training data to tensor
    X_train_tensor = torch.tensor(train_data[:, :-1], dtype=torch.float32)  # Input (hour, region, duration)
    y_train_tensor = torch.tensor(train_data[:, -1], dtype=torch.float32).unsqueeze(1)  # Target (is_present)
    print("X_train_tensor Shape:", X_train_tensor.shape)  # Should be (batch_size, seq_len, 4)

    # Reshape input for LSTM (batch_size, sequence_length=1, features)
    X_train_tensor = X_train_tensor.unsqueeze(1)

    criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(X_train_tensor)

        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"✅ Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "../elderly_movement_predictor.pth")
    print("✅ Model training complete and saved.")
    return model


# ==============================
# 4️⃣ Detect Anomalies (Missing Presence)
# ==============================
def detect_anomalies(test_data, model, threshold=0.4):
    """
    Detect anomalies in hour-region presence patterns.
    """
    model.eval()

    # Convert test data to tensor
    X_test_tensor = torch.tensor(test_data, dtype=torch.float32)
    print(f"Shape of test data before reshape: {X_test_tensor.shape}")

    # Reshaping for LSTM
    X_test_tensor = X_test_tensor.unsqueeze(1)  # Reshape for LSTM (check if this is correct)
    print(f"Shape after unsqueeze: {X_test_tensor.shape}")

    with torch.no_grad():
        try:
            predictions = model(X_test_tensor)
        except Exception as e:
            print(f"Error during model prediction: {e}")
            raise

    # Normalize scores using sigmoid
    anomaly_scores = torch.sigmoid(predictions).numpy().flatten()

    # Anomalies are data points where the score is above threshold
    anomalies = anomaly_scores > threshold

    return anomalies, anomaly_scores


# ==============================
# 5️⃣ Run Training & Testing
# ==============================
def main(train_file, test_file, num_epochs=50, threshold=0.4):
    """
    Train the model using 'train_file' and detect anomalies in 'test_file'.
    """

    # Load training data
    print("🔹 Loading training data...")
    df_train, train_data, scaler = load_and_preprocess_data(train_file)
    print("Train Data Shape:", train_data.shape)  # Should be (num_samples, 4)

    # Train the model
    print("🔹 Training the model...")
    model = LSTMForecastingModel(input_size=2)
    trained_model = train_model(train_data, model, num_epochs=num_epochs)

    # Load test data
    print("🔹 Loading test data...")
    df_test, test_data, _ = load_and_preprocess_data(test_file)  # Use same scaler if needed
    print("Test Data Shape:", test_data.shape)  # Should be (num_samples, 4)

    # Detect anomalies
    print("🔹 Detecting anomalies...")
    anomalies, anomaly_scores = detect_anomalies(test_data[:, 1:], trained_model, threshold=threshold)

    # Add anomaly detection results to DataFrame
    df_test["anomaly_score"] = anomaly_scores
    df_test["is_anomaly"] = anomalies

    # Display detected anomalies
    anomalies_df = df_test[df_test["is_anomaly"] == True]
    print("\n🚨 Detected Anomalies:")
    print(anomalies_df[["date", "hour", "region", "anomaly_score"]])
    # Calculate percentage of anomalies
    total_test_data = len(df_test)
    total_anomalies = len(anomalies_df)
    anomaly_percentage = (total_anomalies / total_test_data) * 100
    print(f"\n📊 Percentage of anomalies: {anomaly_percentage:.2f}%")
    # Plot anomaly scores
    plt.figure(figsize=(10, 5))
    plt.plot(anomaly_scores, label="Anomaly Score")
    plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold")
    plt.xlabel("Test Data Point")
    plt.ylabel("Probability of Presence")
    plt.legend()
    plt.title("Anomaly Detection Scores (Test Data)")
    plt.savefig("anomaly_scores.png")

    return df_test

# Function to load the trained model
def load_trained_model(model_path="elderly_movement_predictor.pth"):
    model = LSTMForecastingModel(input_size=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
def check_data_for_nans(data):
    if np.isnan(data).any():
        print("⚠️ Warning: Data contains NaN values!")
        print(pd.DataFrame(data).isna().sum())
    else:
        print("✅ No NaN values found in the dataset.")


def plot_elderly_movement(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Set up plot
    plt.figure(figsize=(12, 6))

    # Get unique day indexes
    unique_days = df["day_index"].unique()

    # Define color map
    colors = [plt.colormaps["tab10"](i % 10) for i in range(len(unique_days))]

    # Plot each day_index as a separate line
    for i, day in enumerate(unique_days):
        subset = df[df["day_index"] == day]
        plt.plot(subset["hour"], subset["region"], marker='o', linestyle='-', color=colors[i], label=f"Day {day}")

    # Labels and legend
    plt.xlabel("Hour")
    plt.ylabel("Region")
    plt.title("Elderly Movement by Hour and Region")
    plt.xticks(range(24))  # Show all hours on the x-axis
    plt.yticks(df["region"].unique())  # Ensure all regions are marked on y-axis
    plt.legend(title="Day Index", loc="upper left", bbox_to_anchor=(1, 1))  # Place legend outside
    plt.grid(True, linestyle="--", alpha=0.5)

    # Show plot
    plt.savefig("test_movement.png")

# ==============================
# 6️⃣ Run the Full Pipeline
# ==============================
if __name__ == "__main__":
    train_file = "synthetic_elderly_movement.csv"  # Change this to your training dataset file
    test_file = "test_data.csv"  # Change this to your test dataset file
    num_epochs = 50
    threshold = 0.4  # Higher threshold = more strict anomaly detection

    plot_elderly_movement("test_data.csv")

    main(train_file,test_file,num_epochs,threshold)

    # # Load training data
    # print("🔹 Loading training data...")
    # df_train, train_data, scaler = load_and_preprocess_data(train_file)
    # print("Train Data Shape:", train_data.shape)  # Should be (num_samples, 4)
    # check_data_for_nans(train_data)
    #
    # # Train the model
    # print("🔹 Training the model...")
    # model = LSTMForecastingModel(input_size=2)
    # trained_model = train_model(train_data, model, num_epochs=num_epochs)
    #
    # # Load test data
    # print("🔹 Loading test data...")
    # df_test, test_data, _ = load_and_preprocess_data(test_file)  # Use same scaler if needed
    # print("Test Data Shape:", test_data.shape)  # Should be (num_samples, 4)
    #
    # # Detect anomalies
    # print("🔹 Detecting anomalies...")
    # anomalies, anomaly_scores = detect_anomalies(test_data[:, 1:], trained_model, threshold=threshold)
