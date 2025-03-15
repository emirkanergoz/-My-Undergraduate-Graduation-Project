import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Pandas seçeneklerini ayarla (tüm sütunları ve satırları göster)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

# Veri setlerini yükle
df = pd.read_parquet("../vpn/new_train_data.parquet")
attacker_continent = pd.read_parquet("attacker_continent.parquet")
watcher_continent = pd.read_parquet("watcher_continent.parquet")
print(df.head())

# --- ATTACK TIME İşlemleri ---
# Attack_time sütununu tarih ve saat bileşenlerine ayır
df['tarih'] = pd.to_datetime(df['attack_time']).dt.date
df['saat'] = pd.to_datetime(df['attack_time']).dt.hour

# Saat dilimine göre zaman kategorisi atayan fonksiyon
def zaman_dilimi(saat):
    if 6 <= saat < 12:
        return "Sabah"
    elif 12 <= saat < 18:
        return "Öğlen"
    elif 18 <= saat < 24:
        return "Akşam"
    else:
        return "Gece"

df['zaman_dilimi'] = df['saat'].apply(zaman_dilimi)

# Zaman dilimlerine göre label dağılımı
zaman_dilimi_distribution = df.groupby(['zaman_dilimi', 'label']).size().unstack(fill_value=0)

# VPN olan atakların tüm ataklara oranı
zaman_dilimi_distribution['label_1_ratio'] = zaman_dilimi_distribution[1] / (
    zaman_dilimi_distribution[0] + zaman_dilimi_distribution[1]
)

# Zaman dilimine göre oran haritası oluştur
label_1_ratio_map = {
    "Akşam": 0.056,
    "Gece": 0.053,
    "Sabah": 0.058,
    "Öğlen": 0.053,
}

# Gereksiz sütunları kaldır ve zaman dilimi oranlarını ekle
df.drop("attack_time", axis=1, inplace=True)
df['attack_time'] = df['zaman_dilimi'].map(label_1_ratio_map)
df.drop(columns=["tarih", "saat", "zaman_dilimi"], axis=1, inplace=True)

# --- Eksik Veri Doldurma ---
# watcher_country sütunundaki eksik değerleri en sık görülen değer (mod) ile doldur
wc_mode = df["watcher_country"].mode()[0]
df["watcher_country"] = df["watcher_country"].fillna(wc_mode)

# --- Gereksiz Sütunları Kaldır ---
df.drop(columns=["watcher_as_name", "attacker_as_name"], axis=1, inplace=True)

# --- attack_type One-Hot Encoding ---
attack_type = pd.get_dummies(df["attack_type"], dtype=int)
df = pd.concat([df, attack_type], axis=1)

# --- Etiket (Label) Ayırma ---
label = df["label"]
df.drop(columns=["label", "attack_type"], axis=1, inplace=True)
df = pd.concat([df, label], axis=1)

# --- Eksik Değerleri Doldurma ---
# attacker_as_num sütunundaki eksik değerleri ortalama ile doldur
aan_mean = df["attacker_as_num"].mean()
df["attacker_as_num"] = df["attacker_as_num"].fillna(aan_mean)

# --- Özellikler ve Etiket Ayrımı ---
X = df.drop(columns=["label"])
y = df["label"]

# --- Kullanılmayan Sütunları Kaldır ---
df.drop(columns=["attacker_country"], inplace=True)
df.drop(columns=["watcher_country"], inplace=True)

# --- Kıta Bilgilerini Ekleyerek Veri Setini Güncelle ---
df = pd.concat([df, attacker_continent, watcher_continent], axis=1)
df = df[[*df.columns.difference(['label']), 'label']]

# --- MinMaxScaler ile Ölçekleme ---
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# --- watcher_as_num ve watcher_uuid_enum Sütunlarını Yeniden Konumlandır ---
col = df_scaled.pop("watcher_as_num")
col2 = df_scaled.pop("watcher_uuid_enum")

df_scaled.insert(3, "watcher_as_num", col)
df_scaled.insert(4, "watcher_uuid_enum", col2)

df_scaled.to_parquet("Clean_Vpn_Attack.parquet")
print(df_scaled.head())
