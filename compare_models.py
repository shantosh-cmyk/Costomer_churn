# compare_models.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset
df = pd.read_csv("dataset.csv")

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Drop customerID
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# Encode categorical features
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    if col != "Churn":
        df[col] = LabelEncoder().fit_transform(df[col])

# Normalize and encode target column
df["Churn"] = df["Churn"].astype(str).str.strip().str.lower()
df["Churn"] = df["Churn"].map({"yes": 1, "no": 0})
df.dropna(subset=["Churn"], inplace=True)

print("Dataset shape after cleaning:", df.shape)
print("Churn value counts:\n", df["Churn"].value_counts(dropna=False))

# Split dataset
X = df.drop("Churn", axis=1).values
y = df["Churn"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Classical Models ---
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

svm = SVC(class_weight="balanced", kernel="rbf")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# --- Neural Network ---
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

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

model = ChurnNN(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30
for epoch in range(epochs):
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    y_pred_nn = model(X_test_t)
    y_pred_nn_class = (y_pred_nn >= 0.5).int().numpy().flatten()

# --- Collect Metrics ---
results = {
    "Logistic Regression": [
        accuracy_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_lr)
    ],
    "Random Forest": [
        accuracy_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_rf)
    ],
    "SVM": [
        accuracy_score(y_test, y_pred_svm),
        precision_score(y_test, y_pred_svm),
        recall_score(y_test, y_pred_svm),
        f1_score(y_test, y_pred_svm)
    ],
    "Neural Network": [
        accuracy_score(y_test, y_pred_nn_class),
        precision_score(y_test, y_pred_nn_class),
        recall_score(y_test, y_pred_nn_class),
        f1_score(y_test, y_pred_nn_class)
    ]
}

# Display results as table
df_results = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score"])
print("\nModel Comparison:\n")
print(df_results.T)