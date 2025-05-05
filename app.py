from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

# FastAPI uygulamasını başlat
app = FastAPI()

# Modeli yükleme
model = joblib.load('random_forest_model.joblib')


# Giriş verisini almak için Pydantic modelini oluşturma
class InputData(BaseModel):
    attack_time: float
    attacker_as_num: float
    attacker_ip_enum: float
    watcher_as_num: float
    watcher_uuid_enum: float
    continent_attack_Africa: float
    continent_attack_Asia: float
    continent_attack_Europe: float
    continent_attack_North_America: float
    continent_attack_Oceania: float
    continent_attack_South_America: float
    continent_watcher_Africa: float
    continent_watcher_Asia: float
    continent_watcher_Europe: float
    continent_watcher_North_America: float
    continent_watcher_Oceania: float
    continent_watcher_South_America: float
    database_bruteforce: float
    ftp_bruteforce: float
    http_bruteforce: float
    http_crawl: float
    http_exploit: float
    http_scan: float
    http_spam: float
    pop3_imap_bruteforce: float
    sip_bruteforce: float
    smb_bruteforce: float
    ssh_bruteforce: float
    tcp_scan: float
    telnet_bruteforce: float
    unknown_unknown: float
    windows_bruteforce: float


# Tahmin endpoint'i
@app.post("/predict/")
def predict(data: InputData):
    # Veriyi uygun formata dönüştürme
    prediction_data = np.array([[
        data.attack_time,
        data.attacker_as_num,
        data.attacker_ip_enum,
        data.watcher_as_num,
        data.watcher_uuid_enum,
        data.continent_attack_Africa,
        data.continent_attack_Asia,
        data.continent_attack_Europe,
        data.continent_attack_North_America,
        data.continent_attack_Oceania,
        data.continent_attack_South_America,
        data.continent_watcher_Africa,
        data.continent_watcher_Asia,
        data.continent_watcher_Europe,
        data.continent_watcher_North_America,
        data.continent_watcher_Oceania,
        data.continent_watcher_South_America,
        data.database_bruteforce,
        data.ftp_bruteforce,
        data.http_bruteforce,
        data.http_crawl,
        data.http_exploit,
        data.http_scan,
        data.http_spam,
        data.pop3_imap_bruteforce,
        data.sip_bruteforce,
        data.smb_bruteforce,
        data.ssh_bruteforce,
        data.tcp_scan,
        data.telnet_bruteforce,
        data.unknown_unknown,
        data.windows_bruteforce
    ]])

    # Tahmin yapma
    prediction = model.predict(prediction_data)

    # Tahmin sonucunu döndürme
    return {"prediction": int(prediction[0])}