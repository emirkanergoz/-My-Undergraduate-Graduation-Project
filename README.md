# VPN Kullanımını Tespit Etme - Siber Saldırı Analizi

Bu proje, siber saldırıların VPN kullanarak yapılıp yapılmadığını tespit etmeye yönelik olarak geliştirilmiştir. Makine öğrenimi (ML) ve derin öğrenme (DL) algoritmaları ile analiz gerçekleştirilmiştir.

## Proje İçeriği

## 📁 Dosya Yapısı

- `new_train_data.parquet`: Orijinal ham veri seti
- `CleanData.py`: Veri ön işleme ve kıta bazlı ayırma işlemleri
- `Machine_Learning.py`: ML ve DL algoritmalarının uygulandığı ana dosya
- `attacker_continent.parquet`: Saldırganların kıta bilgileri
- `watcher_continent.parquet`: İzleyicilerin kıta bilgileri
- `Attacker_Country.py`: Saldırgan ülkeleri kıtalara ayırır
- `Watcher_Country.py`: İzleyici ülkeleri kıtalara ayırır
- `Clean_Vpn_Attack.parquet`: Ön işleme sonrası oluşan temiz veri seti
- `app.py: FastAPI uygulamasını çalıştırır
- `joblibModelKaydetme.py: Random Forest modelinin joblib ile kaydeder
- `random_forest_model.joblib: Kaydedilen Random Forest modeli

## Kullanım
1. Orijinal veri seti olan `new_train_data.parquet` dosyasını kullanarak veri ön işleme işlemini başlatın:
   ```bash
   python CleanData.py

   Bu işlem sonunda Clean_Vpn_Attack.parquet dosyası oluşacaktır.

   Ardından makine öğrenimi ve derin öğrenme algoritmalarını çalıştırmak için:
   python Machine_Learning.py

   Not: Sadece analiz sonuçlarını görmek istiyorsanız, doğrudan Clean_Vpn_Attack.parquet ve Machine_Learning.py dosyalarını kullanabilirsiniz.
   Ön işleme ve diğer süreçlere gerek yoktur. Bu dosyalar, modelin sonuçlarına doğrudan erişim sağlar.

   FastAPI uygulamasını başlatmak için:
   Modelin bulunduğu konuma gidip şu kodu çalıştırın:
   uvicorn app:app --reload
   Daha sonra "http://127.0.0.1:8000/docs" adresine gidin. "Try it out" butonuna basıp örnek veriler girerek tahmin sonuçlarını kontrol edebilirsiniz.
   Verinin tahmin sonucu şu şekilde görünecektir:
   {
  "prediction": 0
   }
   
## Kullanılan Algoritmalar

**Makine Öğrenimi:**
- Lojistik Regresyon
- KNN (En Yakın Komşu)
- Karar Ağacı
- Rastgele Orman
- Doğrusal Regresyon

**Derin Öğrenme:**
- MLP (Multi-Layer Perceptron)

## 📊 Sonuçlar

Modellerin başarıları, aşağıdaki metriklerle değerlendirilmiştir:
- Doğruluk (Accuracy)
- Precision
- Recall
- F1-Score

## Model Seçimi ve Değerlendirme
Bu projede, çeşitli makine öğrenimi algoritmaları uygulanmış ve en iyi sonuçları veren model olarak Random Forest belirlenmiştir. Ayrıca, modelin aşırı öğrenme (overfitting) yapıp yapmadığını kontrol etmek amacıyla Cross Validation uygulanmış ve modelin aşırı öğrenme yapmadığı doğrulanmıştır. Bu sayede daha güvenilir ve genellenebilir bir model elde edilmiştir.

### Örnek Görseller:
- Confusion Matrix
- Başarı Oranı Grafikleri
- Random Forest F1 skoru ve ROC AUC skorları.
  
### Gereksinimler:
Proje Python 3.x ile çalışmaktadır. Gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanabilirsiniz:

```bash
pip install -r requirements.txt
