# nn_churn.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("dataset.csv")

# --- Debug raw Churn values ---
print("Unique values in Churn column (raw):", df["Churn"].unique())

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    if col != "Churn":  # don't encode target here
        df[col] = LabelEncoder().fit_transform(df[col])

df["Churn"] = df["Churn"].astype(str).str.strip().str.lower()
print("Unique values after normalization:", df["Churn"].unique())

df["Churn"] = df["Churn"].map({"yes": 1, "no": 0})
df.dropna(subset=["Churn"], inplace=True)

print("Dataset shape after cleaning:", df.shape)
print("Churn value counts:\n", df["Churn"].value_counts(dropna=False))

X = df.drop("Churn", axis=1).values
y = df["Churn"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class ChurnNN(nn.Module):
    def __init__(self, input_dim):
        super(ChurnNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize model
model = ChurnNN(X_train.shape[1])

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred >= 0.5).int()

# Convert to numpy for metrics
y_true = y_test.numpy()
y_pred_class = y_pred_class.numpy()

print("\nNeural Network Results:")
print("Accuracy:", accuracy_score(y_true, y_pred_class))
print("Precision:", precision_score(y_true, y_pred_class))
print("Recall:", recall_score(y_true, y_pred_class))

print("F1 Score:", f1_score(y_true, y_pred_class))
