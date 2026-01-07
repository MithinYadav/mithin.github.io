import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load Dataset (Simulated based on your project description)
# In your repo, replace this with your actual CSV file
data = pd.read_csv('customer_data.csv') 

# 2. Data Cleaning [cite: 12]
data = data.dropna() # Removing missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# 3. Basic Predictive Model [cite: 13]
X = data[['Tenure', 'MonthlyCharges', 'TotalCharges']] 
y = data['Churn'] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print(f"Model Training Complete. Accuracy: {model.score(X_test, y_test)}")
