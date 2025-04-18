import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from sklearn.ensemble import RandomForestClassifier

# Use a model that supports probability prediction
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Load dataset
df = pd.read_csv('Crop_recommendation.csv')  # Change filename if needed

# Verify column names
print(df.columns)

# Set target column correctly
target_column = 'label'  # Change based on your dataset
X = df.drop(columns=[target_column])  
y = df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
ms = MinMaxScaler()
sc = StandardScaler()

X_train_scaled = ms.fit_transform(X_train)
X_train_final = sc.fit_transform(X_train_scaled)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train_final, y_train)

# Save model and scalers
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('minmaxscaler.pkl', 'wb') as f:
    pickle.dump(ms, f)
with open('standscaler.pkl', 'wb') as f:
    pickle.dump(sc, f)

print("Model and scalers retrained & saved successfully!")
