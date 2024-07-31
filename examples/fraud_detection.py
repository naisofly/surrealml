import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from surrealml import SurMlFile, Engine

# Step 1: Load and preprocess the data
df = pd.read_csv('creditcard.csv')
df = df[['Time', 'Amount', 'Class']]  # Keep only Time, Amount, and Class

X = df[['Time', 'Amount']]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 3: Evaluate the model
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Step 4: Convert the model to ONNX format
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Step 5: Save the ONNX model
with open("credit_card_fraud_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

    # Step 6: Create a SurML file
file = SurMlFile(
    model=onnx_model,
    name="credit_card_fraud_detection",
    inputs=X_train.values,
    engine=Engine.ONNX
)

file.add_version(version="0.0.1")
file.add_column("Time")
file.add_column("Amount")

# Step 7: Save the SurML file
file.save(path="./credit_card_fraud_model.surml")

# Step 8: Upload the model to SurrealDB (adjust credentials as needed)
SurMlFile.upload(
    path="./credit_card_fraud_model.surml",
    url="http://127.0.0.1:8000/ml/import",
    chunk_size=36864,
    namespace="test",
    database="test",
    username="root",
    password="root"
)

# Step 9: Now you can use the model in SurrealQL!