import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_parquet("Clean_Vpn_Attack.parquet")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)


X = df.drop(columns=['label'])
y = df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 1. Doğrusal Regresyon (Linear Regression)
print("\nDoğrusal Regresyon Uygulaması:")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_train_pred_lr = linear_model.predict(X_train)
y_val_pred_lr = linear_model.predict(X_val)
y_test_pred_lr = linear_model.predict(X_test)

print("Eğitim Seti Başarısı (Doğrusal Regresyon):", accuracy_score(y_train, np.round(y_train_pred_lr))*100)
print("Validation Seti Başarısı (Doğrusal Regresyon):", accuracy_score(y_val, np.round(y_val_pred_lr))*100)
print("Test Seti Başarısı (Doğrusal Regresyon):", accuracy_score(y_test, np.round(y_test_pred_lr))*100)

cm = confusion_matrix(y_test, np.round(y_test_pred_lr))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Etiket')
plt.title('Confusion Matrix')
plt.show()

# 2. Karar Ağaçları (Decision Trees)
print("\nKarar Ağaçları Uygulaması:")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_train_pred_dt = dt_model.predict(X_train)
y_val_pred_dt = dt_model.predict(X_val)
y_test_pred_dt = dt_model.predict(X_test)

# Başarıları % olarak yazdır
print("Eğitim Seti Başarısı (Karar Ağaçları):", accuracy_score(y_train, y_train_pred_dt) * 100)
print("Validation Seti Başarısı (Karar Ağaçları):", accuracy_score(y_val, y_val_pred_dt) * 100)
print("Test Seti Başarısı (Karar Ağaçları):", accuracy_score(y_test, y_test_pred_dt) * 100)

# Confusion Matrix
conf_matrix_dt = confusion_matrix(y_test, y_test_pred_dt)
sns.heatmap(conf_matrix_dt, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix - Karar Ağaçları')
plt.show()

# En önemli özellikleri göster
feature_importances_dt = pd.DataFrame(dt_model.feature_importances_, index=X.columns, columns=['Importance'])
print("En Önemli Özellikler (Karar Ağaçları):")
print(feature_importances_dt.sort_values(by='Importance', ascending=False).head())

# 3. K-En Yakın Komşu (K-Nearest Neighbors - KNN)
print("\nK-En Yakın Komşu Uygulaması:")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_train_pred_knn = knn_model.predict(X_train)
y_val_pred_knn = knn_model.predict(X_val)
y_test_pred_knn = knn_model.predict(X_test)

# Başarıları % olarak yazdır
print("Eğitim Seti Başarısı (KNN):", accuracy_score(y_train, y_train_pred_knn) * 100)
print("Validation Seti Başarısı (KNN):", accuracy_score(y_val, y_val_pred_knn) * 100)
print("Test Seti Başarısı (KNN):", accuracy_score(y_test, y_test_pred_knn) * 100)

# Confusion Matrix
conf_matrix_knn = confusion_matrix(y_test, y_test_pred_knn)
sns.heatmap(conf_matrix_knn, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix - K-En Yakın Komşu (KNN)')
plt.show()

# 4. Rastgele Orman (Random Forest)
print("\nRastgele Orman Uygulaması:")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_train_pred_rf = rf_model.predict(X_train)
y_val_pred_rf = rf_model.predict(X_val)
y_test_pred_rf = rf_model.predict(X_test)

# Başarıları % olarak yazdır
print("Eğitim Seti Başarısı (Rastgele Orman):", accuracy_score(y_train, y_train_pred_rf) * 100)
print("Validation Seti Başarısı (Rastgele Orman):", accuracy_score(y_val, y_val_pred_rf) * 100)
print("Test Seti Başarısı (Rastgele Orman):", accuracy_score(y_test, y_test_pred_rf) * 100)

# Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test, y_test_pred_rf)
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix - Rastgele Orman')
plt.show()

# En önemli özellikleri göster
feature_importances_rf = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['Importance'])
print("En Önemli Özellikler (Rastgele Orman):")
print(feature_importances_rf.sort_values(by='Importance', ascending=False).head())



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veriyi eğitim, doğrulama ve test setlerine böl (70-10-20)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# PyTorch tensörlerine çevir
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Veri yükleyicileri oluştur
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Model tanımla
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)  # Eğer çok sınıflıysa, 1 yerine sınıf sayısı koy
        self.sigmoid = nn.Sigmoid()  # Çok sınıflıysa, Softmax kullan

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return self.sigmoid(x)  # Çok sınıflıysa: return torch.softmax(x, dim=1)


# Modeli oluştur
model = MLP(input_size=32)
criterion = nn.BCELoss()  # Çok sınıflıysa: nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Modeli eğit
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)  # BCE için uygun format
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

# Modeli değerlendirme
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor.to(device)).cpu().numpy()
    y_test_pred = (y_test_pred > 0.5).astype(int)

from sklearn.metrics import accuracy_score

print("Test Seti Başarısı:", accuracy_score(y_test, y_test_pred) * 100)

y_pred_test = y_test_pred.flatten()
y_true_test = y_test.values.flatten()

# Confusion Matrix'i hesapla
conf_matrix = confusion_matrix(y_true_test, y_pred_test)

# Confusion Matrix'i görselleştir
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()

# Sınıflandırma Raporu
print("\nClassification Report:")
print(classification_report(y_true_test, y_pred_test))


# Lojistik Regresyon

# Logistic Regression modelini tanımla
logreg_model = LogisticRegression(class_weight='balanced', max_iter=1000)

# Modeli eğit
logreg_model.fit(X_train, y_train)

# Tahminler
y_train_pred_lr = logreg_model.predict(X_train)
y_val_pred_lr = logreg_model.predict(X_val)
y_test_pred_lr = logreg_model.predict(X_test)

# Başarı skorları
print("Eğitim Seti Başarısı (Logistic Regression):", accuracy_score(y_train, y_train_pred_lr)*100)
print("Validation Seti Başarısı (Logistic Regression):", accuracy_score(y_val, y_val_pred_lr)*100)
print("Test Seti Başarısı (Logistic Regression):", accuracy_score(y_test, y_test_pred_lr)*100)

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred_lr)

# Confusion matrix'i görselleştir
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Etiket')
plt.title('Confusion Matrix (Logistic Regression)')
plt.show()