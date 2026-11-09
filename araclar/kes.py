import cv2, sys, os, glob, shutil
KLASOR = r"C:\Users\saru2\Videos"
CIKIS  = os.path.join(KLASOR, "kalibrasyon.mp4")
BASLA_DK, SURE_DK = 5, 5

# en buyuk mp4 = uzun kayit (YENİAÇI.mp4)
GIRIS = max(glob.glob(os.path.join(KLASOR, "*.mp4")), key=os.path.getsize)
print("kaynak:", GIRIS, f"({os.path.getsize(GIRIS)//2**20} MB)")

cap = cv2.VideoCapture(GIRIS)
if not cap.isOpened():
    # OpenCV Turkce karakterli yolu acamiyor -> ASCII adla gecici kopya
    gecici = os.path.join(KLASOR, "_gecici.mp4")
    print("Turkce karakter sorunu, gecici kopya aliniyor (1-2 dk)...")
    shutil.copyfile(GIRIS, gecici)
    cap = cv2.VideoCapture(gecici)
    if not cap.isOpened():
        sys.exit("HATA: kopya da acilamadi -> video formati desteklenmiyor olabilir")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
print(f"video acildi: {int(cap.get(7))} kare, {fps:.0f} fps")
cap.set(cv2.CAP_PROP_POS_MSEC, BASLA_DK * 60_000)
out = cv2.VideoWriter(CIKIS, cv2.VideoWriter_fourcc(*"mp4v"), fps / 2, (1280, 720))
yazilan = 0
for i in range(int(SURE_DK * 60 * fps)):
    ok, kare = cap.read()
    if not ok: break
    if i % 2 == 0:
        out.write(cv2.resize(kare, (1280, 720))); yazilan += 1
    if i % 900 == 0: print(f"  {i // int(fps)} sn...")
out.release(); cap.release()
print(f"bitti: {yazilan} kare yazildi -> {CIKIS}")