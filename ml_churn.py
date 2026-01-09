# ml_churn.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("dataset.csv")

# 2. Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# Fill missing values with median instead of dropping everything
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# 3. Drop customerID (not useful for prediction)
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# 4. Encode categorical features
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# 5. Encode target column (Churn: Yes/No → 1/0)
if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop rows where Churn is NaN (if any)
df.dropna(subset=["Churn"], inplace=True)

# Debug check
print("Dataset shape after cleaning:", df.shape)
print("Churn value counts:\n", df["Churn"].value_counts(dropna=False))

# 6. Split dataset
X = df.drop("Churn", axis=1)
y = df["Churn"]

if len(df) == 0:
    raise ValueError("Dataset is empty after cleaning. Please check your CSV file contents.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Scale features (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train models
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

svm = SVC(class_weight="balanced", kernel="rbf")
svm.fit(X_train, y_train)

# 8. Evaluate models
def evaluate(model, name):
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

evaluate(log_reg, "Logistic Regression")
evaluate(rf, "Random Forest")
evaluate(svm, "SVM")