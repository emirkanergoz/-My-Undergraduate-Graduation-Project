# VPN Detection on Cyber Attack Data

Bu proje, siber saldırıların VPN kullanıp kullanmadığını tespit etmek amacıyla geliştirilmiştir. Veri seti, "new_train_data.parquet" adlı dosyada bulunan çeşitli özellikler kullanılarak işlenmiş ve makine öğrenimi (ML) ile derin öğrenme (DL) algoritmaları uygulanmıştır.

## Proje İçeriği

### Dosyalar:
1. **new_train_data.parquet**: Orijinal veri seti.
2. **CleanData.py**: Veri ön işleme süreci ve saldırgan/izleyici verilerinin kıtalara ayrılması işlemleri.
3. **Machine_Learning.py**: Makine öğrenimi ve derin öğrenme algoritmalarının uygulandığı Python dosyası.
4. **attacker_continent.parquet**: Saldırgan kıtaları bilgilerini içeren parquet dosyası.
5. **watcher_continent.parquet**: İzleyici kıtaları bilgilerini içeren parquet dosyası.
6. **Attacker_Country.py**: Saldırgan ülke bilgilerini kıtalara ayıran Python dosyası.
7. **Watcher_Country.py**: İzleyici ülke bilgilerini kıtalara ayıran Python dosyası.
8. **Clean_Vpn_Attack.parquet**: Temizlenmiş ve işlenmiş veri seti.


## Kullanım
Orjinal veri setini aldıktan sonra,(new_train_data.parquet) CleanData.py dosyası ile veri ön işleme sürecini gerçekleştirebilirsiniz. Veri ön işleme süreci bittikten sonra Clean_Vpn_Attack.parquet dosyası oluşacaktır. Machine_Learning.py dosyası ile temizlenmiş veri seti üzerine 5 adet Makine Öğrenimi algoritmalarını ve 1 adet Derin Öğrenme Algoritmasını uygulayabilirsiniz.

## Kullanılan Algoritmalar
Lojistik Regresyon
KNN-En Yakın Komşu
Karar Ağacı
Rastgele Orman
Doğrusal Regresyon

Derin Öğrenme Algoritması (MLP)

## Sonuçlar
Makine öğrenimi ve derin öğrenme algoritmaları ile elde edilen sonuçlar, modelin doğruluğu, precision, recall ve F1 skorları gibi metriklerle birlikte görseller halinde sunulmuştur.

Örnek Görseller:
Confusion Matrix
Başarı Oranı

## Not
Sadece Clean_Vpn_Attack.parquet dosyasını ve Machine_Learning.py dosyasını almak ve çalıştırmak isteyenler için, ön işleme ve diğer süreçlere gerek yoktur. Bu dosyalar, modelin sonuçlarına doğrudan erişim sağlar.


### Gereksinimler:
Proje Python 3.x ile çalışmaktadır. Gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanabilirsiniz:

```bash
pip install -r requirements.txt
