"""Uzun kayittan SILIKON SILME anlarinin karelerini cikarir (spatula etiketlemek icin).

Mevcut OP230 modeliyle videoyu seyrek tarar (saniyede birkac kare), her cevrimde
musur sikma / musur takili / silikon silme goruldugu anlari bulur ve o anlarin
etrafindaki kareleri bir klasore JPG olarak yazar. Bu klasoru FrameForge'a yukle,
'spatula' sinifini etiketle.

Kullanim (gozlukv klasorunde):
    python silikon_kare_cikar.py                          -> Videos'taki en buyuk mp4 (YENİAÇI.mp4)
    python silikon_kare_cikar.py "C:\\...\\video.mp4"      -> belirli bir video
Secenekler (dosyanin basinda): TARAMA_FPS, CIKTI_FPS, ONCE_SN, SONRA_SN
"""
import glob, os, sys, time
import cv2

KLASOR = os.path.dirname(os.path.abspath(__file__))
MODEL_OV = os.path.join(KLASOR, "op230_model_1_best_openvino_model")
MODEL_PT = os.path.join(KLASOR, "op230_model_1_best.pt")
CIKTI = os.path.join(KLASOR, "silikon_kareler")

TARAMA_FPS = 3.0     # videoyu saniyede kac kare tarayalim (hiz icin seyrek)
CIKTI_FPS = 2.0      # bulunan anlardan saniyede kac kare yazilsin
ONCE_SN = 2.0        # tetikleyici andan bu kadar once basla
SONRA_SN = 6.0       # son tetikleyiciden bu kadar sonrasina kadar yaz (silikon musurden sonra gelir)
BIRLESTIR_SN = 4.0   # bu kadar yakin anlar tek pencere sayilir
ESIK = {"musur_skma": 0.40, "müşür": 0.35, "silikon_silme": 0.25}


def video_bul():
    if len(sys.argv) > 1:
        return sys.argv[1]
    vids = glob.glob(os.path.join(os.path.expanduser("~"), "Videos", "*.mp4"))
    if not vids:
        sys.exit("HATA: Videos klasorunde mp4 yok. Videoyu arguman olarak ver.")
    return max(vids, key=os.path.getsize)


def main():
    from ultralytics import YOLO
    yol = video_bul()
    cap = cv2.VideoCapture(yol)
    if not cap.isOpened():
        sys.exit(f"HATA: video acilamadi -> {yol}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sure = n / fps
    print(f"video: {yol}\n  {n} kare, {fps:.0f} fps, {sure/60:.1f} dk")

    model = YOLO(MODEL_OV, task="detect") if os.path.isdir(MODEL_OV) else YOLO(MODEL_PT)
    print(f"model: {'OpenVINO' if os.path.isdir(MODEL_OV) else 'PyTorch'}")

    # 1) tarama: tetikleyici anlari bul
    adim = max(1, int(round(fps / TARAMA_FPS)))
    anlar, t0 = [], time.time()
    for k in range(0, n, adim):
        if k > 0:                      # sirali okuma (konum atlamadan hizli):
            for _ in range(adim - 1):  # aradaki kareleri cozmeden gec
                cap.grab()
        ok, kare = cap.read()
        if not ok:
            break
        r = model.predict(kare, imgsz=640, conf=min(ESIK.values()), verbose=False)[0]
        if r.boxes is not None:
            for c, cf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                ad = model.names[int(c)]
                if ad in ESIK and cf >= ESIK[ad]:
                    anlar.append(k / fps)
                    break
        if (k // adim) % 300 == 0:
            gecen = time.time() - t0
            print(f"  tarama {k / fps / 60:5.1f}/{sure / 60:.1f} dk  ({gecen:.0f} sn gecti)", flush=True)

    if not anlar:
        sys.exit("Hic musur/silikon ani bulunamadi.")

    # 2) yakin anlari pencerelere birlestir
    pencereler = []
    for t in anlar:
        if pencereler and t - pencereler[-1][1] <= BIRLESTIR_SN:
            pencereler[-1][1] = t
        else:
            pencereler.append([t, t])
    pencereler = [(max(0.0, a - ONCE_SN), min(sure, b + SONRA_SN)) for a, b in pencereler]
    print(f"{len(pencereler)} pencere bulundu, toplam {sum(b - a for a, b in pencereler):.0f} sn")

    # 3) pencerelerden kareleri yaz
    os.makedirs(CIKTI, exist_ok=True)
    adi = os.path.splitext(os.path.basename(yol))[0].encode("ascii", "ignore").decode() or "video"
    yazilan = 0
    for i, (a, b) in enumerate(pencereler, 1):
        t = a
        while t <= b:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, kare = cap.read()
            if not ok:
                break
            ad = os.path.join(CIKTI, f"{adi}_p{i:02d}_{t:07.1f}s.jpg")
            ok_, buf = cv2.imencode(".jpg", kare, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok_:
                buf.tofile(ad)          # Turkce karakterli yolda da calisir
                yazilan += 1
            t += 1.0 / CIKTI_FPS
    with open(os.path.join(CIKTI, "pencereler.csv"), "w", encoding="utf-8") as f:
        f.write("pencere,baslangic_sn,bitis_sn\n")
        for i, (a, b) in enumerate(pencereler, 1):
            f.write(f"{i},{a:.1f},{b:.1f}\n")
    print(f"\nbitti: {yazilan} kare -> {CIKTI}")
    print("Simdi bu klasoru FrameForge'a yukle ve 'spatula' sinifini etiketle.")


if __name__ == "__main__":
    main()
