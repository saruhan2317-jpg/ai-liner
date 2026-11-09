# AI-LINER

OYAK HORSE Bursa motor montaj hattında yapay zeka ile kalite takibi.
Geliştirenler: Saruhan Güven, Gökhan Uçak.

## Dosyalar
| Dosya | Açıklama |
|---|---|
| `ai_liner8.py` | Güncel sürüm — OP230 yeni kamera açısı (kep sayımı, müşür, silikon, geç adım toleransı) |
| `ai_liner6.py` | Önceki kararlı sürüm |
| `web_bridge.py`, `web_static/` | Web paneli (bkz. `README_WEB.md`) |
| `araclar/yama_uygula.py` | Windows için yama uygulayıcı |
| `araclar/silikon_kare_cikar.py` | Uzun kayıttan silikon/spatula etiketleme karelerini çıkarır |
| `araclar/kes.py` | Videodan bölüm kesme |

## Modeller (repoda yok, klasöre ayrıca konur)
- `op230_model_1_best.pt` — OP230 ana model (FrameForge ile eğitildi)
- `silikon_best.pt` — isteğe bağlı spatula modeli (varsa otomatik yüklenir)
- `yolov8n-pose.pt` — iskelet modeli

İlk açılışta `.pt` dosyaları OpenVINO'ya otomatik dönüştürülür.

## Çalıştırma
```powershell
& "C:\Program Files\Python312\python.exe" ai_liner8.py
```
