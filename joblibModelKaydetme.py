from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
df = pd.read_parquet("Clean_Vpn_Attack.parquet")

X = df.drop("label", axis=1)
y = df["label"]

# 3. Train-Test Ayrımı ve Ölçekleme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modeli oluşturma
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Modeli eğitme
rf_model.fit(X_train, y_train)

# Eğitim ve test setlerinde doğruluk hesaplama
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

print("Eğitim Seti Başarısı:", accuracy_score(y_train, y_train_pred) * 100)
print("Test Seti Başarısı:", accuracy_score(y_test, y_test_pred) * 100)

# Modeli kaydetme
joblib.dump(rf_model, 'random_forest_model.joblib')

print("Model başarıyla kaydedildi!")

