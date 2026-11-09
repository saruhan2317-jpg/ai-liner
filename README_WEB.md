# AI-LINER — Web Paneli

Mevcut Tkinter uygulamana **dokunmadan**, yanında çalışan modern bir web
panel eklendi. Kamera görüntüsü, motor çevrimi, QA kontrol listesi, KKD
uyumu ve bölgeler artık aynı anda tarayıcıdan da izlenebiliyor —
Tkinter penceresi de her zamanki gibi çalışmaya devam ediyor.

## Nasıl çalışıyor?

- `web_bridge.py` — Flask tabanlı hafif bir sunucu. YOLO/motor/QA
  mantığına hiç dokunmaz; sadece worker thread'in zaten ürettiği
  (görüntü, durum) çıktısını yakalayıp tarayıcıya **MJPEG video** +
  **Server-Sent Events (canlı JSON durum)** olarak yayınlar.
- `ai_liner1.py` içine sadece 4 küçük ekleme yapıldı: içe aktarma,
  `App.__init__` içinde köprüyü başlatma, `_update_display` içinde
  bir satırlık yayın çağrısı, `on_close` içinde temiz kapanış.
- `web_static/` — tarayıcıda açılan arayüz (HTML/CSS/JS, framework yok).

## Kurulum

```bash
pip install flask
```

(Diğer bağımlılıklar zaten mevcuttu: opencv-python, numpy.)

## Çalıştırma

Programı her zamanki gibi başlat:

```bash
python ai_liner1.py
```

Konsolda şu satırı göreceksin:

```
[web] AI-LINER paneli hazir  ->  http://<bilgisayarin-ip-adresi>:8080
```

Aynı ağdaki herhangi bir cihazdan (telefon, tablet, ofisteki başka bir
bilgisayar) bu adresi tarayıcıda açman yeterli. Sadece bu bilgisayardan
erişeceksen `http://localhost:8080` de çalışır.

Portu değiştirmek istersen `ai_liner1.py` içindeki `WEB_PANEL_PORT`
sabitini güncelle.

## Panelden yapılabilenler

- Canlı görüntüyü izleme (MJPEG akışı)
- Motor çevrimi durumu, kayma barı
- QA kontrol listesi (adım kilitleri, ilerleme, bölge/model kaynağı)
- Bölge (ROI) durumları
- KKD (gözlük/kulaklık) uyum oranları
- Sonuç halkası (GEÇTİ / EKSİK / BEKLİYOR) ve sıfırlama geri sayımı
- Kaynak seçimi (webcam / video dosyası / IP kamera), durdurma, QA
  sıfırlama
- Bazı canlı ayarlar (eşikler, opaklık, kare atlama, aç/kapa
  anahtarları) — kaydırıcılar Tkinter'daki aynı değişkenleri günceller

## Sınırlamalar / bilinen notlar

- ROI (bölge) çizimi/düzenleme şu an sadece Tkinter tarafında —
  web paneli bölgeleri sadece **gösterir**, düzenlemez.
- Klip kaydı ve rapor oluşturma butonları web paneline henüz
  taşınmadı; bunlar için Tkinter penceresini kullanmaya devam et.
- `/api/meta` ilk açılışta bağlı webcam'leri kısa süreliğine tarar;
  bu sırada bir kamera zaten Tkinter tarafından kullanılıyorsa nadir
  durumlarda kısa bir çakışma görebilirsin — sayfayı yeniden
  yüklediğinde geçer.
- Flask kurulu değilse program hiçbir hata vermeden, web paneli
  olmadan normal şekilde (sadece Tkinter) çalışmaya devam eder.

## Sonraki adımlar için fikirler

- ROI düzenlemeyi web tarafına taşımak (canvas üzerinde sürükle-bırak)
- Klip oynatma / rapor görüntüleme sekmeleri eklemek
- Birden fazla istasyonu tek panelde yan yana göstermek
