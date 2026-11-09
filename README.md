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

## Modeller (repoda yok, Releases'ten indirilir)
Modeller [Releases](https://github.com/saruhan2317-jpg/ai-liner/releases) sayfasında; indirip `gozlukv` klasörüne koy.

| Release | Dosya | Kod |
|---|---|---|
| [`v8-model-1`](https://github.com/saruhan2317-jpg/ai-liner/releases/tag/v8-model-1) | `op230_model_1_best.pt` (md5 `66c38d9e24b2d0e80e0ff81eb70d99c9`) | `ai_liner8.py` |

- `op230_model_1_best.pt` — OP230 ana model (FrameForge ile eğitildi)
- `silikon_best.pt` — isteğe bağlı spatula modeli (varsa otomatik yüklenir)
- `yolov8n-pose.pt` — iskelet modeli

İlk açılışta `.pt` dosyaları OpenVINO'ya otomatik dönüştürülür.

## Çalıştırma
```powershell
& "C:\Program Files\Python312\python.exe" ai_liner8.py
```
