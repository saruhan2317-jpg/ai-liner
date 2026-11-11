import os
import collections
import copy
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|threads;1"

import sys, time, threading, queue, math, csv, statistics, webbrowser, json
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from typing import Optional
import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO

# NOT: Parmak (mediapipe) takibi kaldirildi — bolgeler bilek/dirsek ile calisir.

# ══ AI-LINER ════════════════════════════════════════════════════════════
# Goruntu Isleme · Kalite · ISG Teknolojisi
PANEL_W    = 340          # sol panel genisligi
TEMA = {                  # koyu / futuristik palet (logodan) — v6: tanitim sayfasiyla ayni "void" tonlari
    "zemin":     "#05080d",
    "panel":     "#0d131c",
    "panel2":    "#090e15",
    "cizgi":     "#18222f",
    "metin":     "#e6edf3",
    "soluk":     "#8f9fb2",
    "cok_soluk": "#5a6b80",
    "cyan":      "#22d3ee",
    "mor":       "#a855f7",
    "yesil":     "#22c55e",
    "sari":      "#f5b301",
    "turuncu":   "#fb923c",
    "kirmizi":   "#ef4444",
}

URUN_ADI   = "AI-LINER"
URUN_ALT   = "Goruntu Isleme · Kalite · ISG Teknolojisi"
SURUM      = "6"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALF = DEVICE == "cuda"
BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))

# ── model dosyalari ──────────────────────────────────────────────────────
# Dosya adlarinda TURKCE KARAKTER KULLANMA — yol sorunu cikarir.
QA_MODEL_PATH     = os.path.join(BASE_DIR, "op230_model_1_best.pt") # v8: FrameForge ile egitilen OP230 modeli
# v8: ISTEGE BAGLI silikon (spatula) modeli. Dosya klasorde YOKSA hicbir sey
# degismez. Varsa YALNIZCA silikon silme beklenirken calisir (musur bitti,
# silikon henuz yok, motor postada) -> normal karelerde ek yuk yok.
SILIKON_MODEL_PATH = os.path.join(BASE_DIR, "silikon_best.pt")
SILIKON_SINIFLAR   = {"spatula", "silikon_silme"}   # bu siniflardan biri motorun ustundeyse silikon sayilir
SILIKON_CONF       = 0.35
SILIKON_PAY        = 0.05      # motor kutusu her yandan bu kadar genisletilir (normalize)
SILIKON_RENK       = (0, 140, 255)   # turuncu (BGR)
# Spatula motora son SILIKON_PENCERE sn icinde en az SILIKON_MIN_KARE karede
# degmeli. Kalibrasyonda kaza dokunuslari 3-4 sn'de en fazla 2 kare; gercek
# silikon cok daha fazla. Sira serbest oldugu icin bu sart yanlis onayi onler.
SILIKON_MIN_KARE   = 3
SILIKON_PENCERE    = 3.0
# v8: ISTEGE BAGLI musur modeli (2 sinif: takili musur + musur sikma). Dosya
# YOKSA hicbir sey degismez. Varsa yalnizca motor postadayken ve musur adimlari
# bitmemisken calisir; tespitleri ana modelinkilerle ayni konum kurallarindan
# gecer (motor govdesi, sikmanin yeri, tani mesajlari).
MUSUR_MODEL_PATH   = os.path.join(BASE_DIR, "musur_best.pt")
MUSUR_MODEL_CONF   = 0.35
MUSUR_MODEL_SINIF  = {                       # yeni modelin sinif adi -> AI-LINER adi
    "musur": "müşür", "müşür": "müşür", "musur_takili": "müşür",
    "musur_sikma": "musur_skma", "musur_skma": "musur_skma",
}
MUSUR_MODEL_RENK   = {"müşür": (60, 200, 120), "musur_skma": (255, 120, 40)}
POSE_MODEL_PATH   = "yolov8n-pose.pt"                              # iskelet

VIDEO_DIR = os.path.join(BASE_DIR, "videos")
LOGO_PATH   = os.path.join(BASE_DIR, "ai_liner_logo.png")     # panel ustu
AMBLEM_PATH = os.path.join(BASE_DIR, "ai_liner_amblem.png")   # rapor basligi
TANITIM_DIR = os.path.join(BASE_DIR, "tanitim")               # kaydirmali tanitim sayfasi (/tanitim/)
IMGSZ_CHOICES = [320, 416, 512, 640, 832, 960, 1280]

DETECTION_THRESHOLD   = 0.70
IP_RECONNECT_ATTEMPTS = 5
IP_RECONNECT_DELAY    = 2.0
POSE_CONF             = 0.25   # iskelet tespit esigi

# ── sinif rolleri ────────────────────────────────────────────────────────
# NOT (v1.5): KKD takibi (gozluk / kulaklik) projeden cikarildi. Model bu
# siniflari uretse bile cizilmez, sayilmaz, panelde gosterilmez.
GIZLI_CLASSES = {"earplug", "goggle", "gozluk"}

# QA kontrol listesi — SADECE islemler.
# NOT: datamatrix okutma artik bu postada yapilmiyor, listeden cikarildi.
QA_STEP_CLASSES = [
    "kam_mili koyma", "kep_sokme", "pim cakma", "musur_skma",
    "silikon_silme",
]

# ── KEP SOKME: MOTOR UZERINDEKI KEPLER (v8) ─────────────────────────────
# Postada tek motor var; kepler silindir kapagi ustunde IKI SIRA halinde.
# Model (op230_model_1_best.pt) her kepi ayri 'kep' kutusu olarak buluyor.
# KEP_HARITA: her kepin MOTOR KUTUSUNA GORE konumu (rx, ry):
#   rx = 0 motorun sol kenari, 1 sag kenari; ry = 0 ust kenar, 1 alt kenar.
# Degerler kalibrasyon.mp4'te modelin 176 karelik 6-kepli tespitlerinden
# (k-means) olculdu. Bir kepin tespitleri merkezinden en fazla ~0.07 sapiyor,
# en yakin iki kep 0.12 uzakta.
KEP_HARITA = [
    # ── arka sira (kameraya uzak) ──
    (0.192, 0.105),   # 1
    (0.373, 0.090),   # 2
    # ── on sira (kameraya yakin) ──
    (0.218, 0.222),   # 3
    (0.406, 0.206),   # 4
    (0.589, 0.185),   # 5
    (0.767, 0.171),   # 6
]
KEP_COUNT           = len(KEP_HARITA)
KEP_SINIF           = "kep"  # modeldeki kep NESNE sinifi (kep_sokme = islem)
KEP_CONF            = 0.40   # kep tespiti icin alt esik
KEP_MATCH_R         = 0.10   # tespit en yakin yuvaya bu mesafe icindeyse atanir
KEP_BOX_PAD         = 0.15   # kep tespiti motor kutusunun bu kadar disina tasabilir
KEP_KAPAK_MAX_RY    = 0.40   # motordaki kep sayilmak icin kutunun ust %40'inda olmali
KEP_STABIL_SECS     = 1.0    # sayilar bu pencerenin MEDYANI ile kararlastirilir
KEP_YOK_SECS        = 2.0    # yuva bu kadar sn hic gorulmezse "sokuldu" sayilir
KEP_REF_MIN_H       = 0.90   # kutu yuksekligi referansin %90'indan kisaysa (kirpilmis) kullanilmaz
# Motor palette DONDURULUNCE sabit harita kayar. Bu yuzden support konumu,
# uzerindeki keplerin GERCEK konumundan ogrenilir (kalibrasyonda kapak
# disindaki keplerin %99'u support kumesinde, kume medyanindan <0.06).
KEP_SUPPORT_YARI    = (0.10, 0.06)   # ogrenilen support merkezinin cevresi (yarim genislik, yukseklik)
KEP_SUPPORT_OGREN   = 0.30           # merkez guncelleme agirligi (0..1)

# ── MUSUR: motor blogu uzerindeki takilma noktasi (v8) ──────────────────
# Model 'müşür' (takili hali) NESNE sinifini buluyor; konum motor kutusuna
# gore kalibrasyon.mp4'te olculdu (157 tespit, medyan).
MUSUR_KONUM          = (0.408, 0.596)
MUSUR_NESNE          = "müşür"
MUSUR_NESNE_CONF     = 0.25
MUSUR_NESNE_R        = 0.12   # musur tespiti bu noktaya bu kadar yakin olmali
MUSUR_MIN_KARE       = 1      # v8: musur nesnesi kalibrasyonda YALNIZCA sikmadan sonra
                              # goruldu -> tek karede gormek yeterli (yanlis onay riski yok)
MUSUR_TAKILI_ADIM    = "musur_takili"   # QA listesindeki sanal adim
# musur_skma ISLEMI yalnizca kutusu musur noktasini ICERIYORSA sayilir.
# (Modelde bu islemin kutusu operator + tork aletini kapsayan buyuk bir kutu;
# merkezi musurden uzakta kaldigi icin mesafe yerine "icerir" kontrolu.)
MUSUR_SKMA_PAD       = 0.02   # kutu kenarina normalize pay
# ...ya da (motor dondurulmus olabilir) sikma kutusunun en az bu orani motor
# govdesinin alt kismiyla (ry >= MUSUR_ALT_RY[0]) ortusmeli. Kalibrasyonda
# gecerli sikma karelerinde 0.14-0.22, gecersizlerde 0.00-0.07.
MUSUR_SKMA_MIN_ORT   = 0.10
# 'müşür' nesnesi motor kutusunun bu dikey araliginda gorulurse sayilir
# (sabit noktaya bakmaz -> motor donse de calisir)
MUSUR_ALT_RY         = (0.35, 0.95)   # musur_skma ortusme kontrolu (motor alt govdesi)
# 'müşür' NESNESI motor kutusu (+pay) icinde herhangi bir yerde gorulurse sayilir
MUSUR_NESNE_RY       = (-0.10, 1.15)
MUSUR_X_PAY          = 0.10   # motor kutusunun yanlarindan tasma payi

# ── GEC ADIM TOLERANSI ──────────────────────────────────────────────────
# Cevrim kapandiktan (motor capadan ciktiktan) sonra bu kadar sn icinde
# gorulen EKSIK adim biten cevrime yazilir (or. silikon, motor cikarken
# silindi). Cevrimin kapanma ani ve suresi DEGISMEZ; sadece sonuc, kayit ve
# EKSIK klibi bu sure kadar bekletilir.
GEC_ADIM_SECS        = 2.0
MUSUR_KONUM_KONTROL  = True
MUSUR_KONUM_SINIFLAR = {"musur_skma", "musur_takma", MUSUR_NESNE}

# Support (kep birakilma) bolgesi — SARI/BEYAZ support MOTOR PALETINE bagli,
# motorla birlikte postaya girip cikiyor. MOTOR KUTUSUNA GORE goreli:
# (x1, y1, x2, y2), x<0 = motorun solu. kalibrasyon.mp4'teki support
# keplerinin %2-%98 araligindan olculdu.
KEP_SUPPORT_REL     = (-0.62, 0.50, -0.03, 0.88)
# Motor hic gorulmediyse kullanilan sabit yedek (normalize ekran koordinati)
KEP_SUPPORT_ROI     = (0.25, 0.70, 0.47, 0.86)

# Kodun kendisinin isaretledigi (modelde sinifi olmayan) QA adimlari
SANAL_ADIMLAR = {MUSUR_TAKILI_ADIM}
QA_STEP_CLASSES.insert(QA_STEP_CLASSES.index("musur_skma") + 1, MUSUR_TAKILI_ADIM)

# KAM MILI KOYMA ileride OTOMATIK (robot) olacak. True yapinca adim QA
# listesinden cikar, cevrim sonucunu etkilemez.
KAM_MILI_OTOMATIK   = False
if KAM_MILI_OTOMATIK:
    QA_STEP_CLASSES = [s for s in QA_STEP_CLASSES if s != "kam_mili koyma"]

# 'kam_mili' NESNESI motorun alt kisminda gorulurse 'kam mili koyma' de
# goruldu sayilir (islem etiketi kacsa bile). Masada/rafta duran kam mili
# sayilmaz: nesne motor kutusunun alt yarisinda, yanlardan KAM_X_PAY icinde
# olmali. Onay suresi ve kopukluk toleransi 'kam_mili koyma' ile ayni.
KAM_NESNE           = "kam_mili"
KAM_NESNE_SAY       = True
KAM_NESNE_CONF      = 0.40
KAM_X_PAY           = 0.15     # motor kutusunun solu/sagi (normalize)
KAM_MIN_RY          = 0.50     # motor kutusunun en az bu yuksekliginden asagida


def support_rect(motor_box):
    """Motor kutusundan (x1,y1,x2,y2 normalize) support bolgesini hesaplar."""
    if motor_box is None:
        return KEP_SUPPORT_ROI
    mx1, my1, mx2, my2 = motor_box
    mw, mh = mx2 - mx1, my2 - my1
    rx1, ry1, rx2, ry2 = KEP_SUPPORT_REL
    c = lambda v: max(0.0, min(1.0, v))
    return (c(mx1 + rx1 * mw), c(my1 + ry1 * mh), c(mx1 + rx2 * mw), c(my1 + ry2 * mh))

# Kapanis adimi yok; cevrim MOTORUN HAREKETIYLE kapanir (asagiya bak).
FINAL_STEP = None
AUTO_RESET_SECS = 1.5      # sonuc ekranda kalma suresi, sonra sifirlanir

# ── CEVRIM KAPATMA: MOTOR HAREKETI ───────────────────────────────────────
# Mantik: motor postaya girip yerlesince "capa" (anchor) noktasi kilitlenir ve
# cevrim baslar. Operator islemleri yapar. Motor capa noktasindan
# MOTOR_MOVE_THRES kadar kayinca (= diger postaya gidiyor) cevrim KAPANIR:
# o ana kadar isaretlenmis adimlara gore GECTI / EKSIK yazilir, AUTO_RESET_SECS
# sonra (ya da yeni motor yerlesir yerlesmez) QA sifirlanir.
#
# ONEMLI — KAYBOLMA CEVRIMI KAPATMAZ:
# Operatorun eli/govdesi motorun onune geldiginde ya da bazi acilarda model
# 'motor' etiketini kaybediyor. Bu yuzden "motor gorunmuyor" ASLA cikis sayilmaz.
# Cevrim yalnizca motor GORULEREK capadan uzaklasirsa kapanir.
MOTOR_CLASS         = "motor"
# YENI ACI (v8): motor postaya YATAY girip cikiyor. Operator motorun ustune
# egilince ya da alt kismi kapaninca kutunun ust/alt kenari 0.2'ye kadar
# oynayabiliyor; bu dikey oynama "motor gitti" sanilmasin diye kayma yalnizca
# YATAY eksende olculur. Kalibrasyon videosu: gercek cikislarda yatay kayma
# >= 0.27, postada itilme en fazla 0.085.
MOTOR_SADECE_YATAY  = True
# Motor kutusunun rengi (BGR). Ultralytics paletinin turkuazi kep/support
# renklerine cok yakindi; panelin mor vurgu rengi (#A855F7) kullaniliyor.
MOTOR_RENK          = (247, 85, 168)
MOTOR_CONF          = 0.35   # motor icin varsayilan conf (arayuzden ayarlanir)
MOTOR_SMOOTH        = 0.40   # merkez yumusatma agirligi (0..1, buyuk = daha cevik)
MOTOR_SETTLE_TOL    = 0.020  # "yerlesti" sayilmak icin izin verilen kayma (normalize)
MOTOR_SETTLE_SECS   = 0.70   # motor bu kadar sure sabit kalirsa capa kilitlenir
MOTOR_MOVE_THRES    = 0.21   # capa noktasindan bu kadar kayarsa CIKIS (arayuzde slider)
MOTOR_MOVE_DEBOUNCE = 0.35   # kayma bu kadar sn kesintisiz surmeli (titreme filtresi)
MOTOR_MOVE_MIN_HITS = 3      # ve en az bu kadar TESPIT karesinde ust uste kaymis olmali
MOTOR_HOLD_SECS     = 12.00   # bu sureye kadar kaybolma yok sayilir (okluzyon toleransi)
MOTOR_ABSENT_CLOSES = False  # VARSAYILAN KAPALI: kaybolma tek basina kapatmaz
MOTOR_LOST_SECS     = 12.0   # yukaridaki acikken, bu kadar surekli yoklukta kapat
MOTOR_SNAP_GAP      = 0.50   # bu kadar yokluktan sonra yeniden gorulurse konum sifirdan alinir
MOTOR_UNDO_SECS     = 1.50   # yanlis kapanma: motor bu sure icinde ayni yere donerse geri al
MOTOR_UNDO_EMPTY_MAX = 0.80  # ...ve posta bu sureden uzun bos kalmamis olmali
                             # (bos kaldiysa parca gercekten gitmis, YENI parcadir)
MOTOR_STATION_R_K   = 1.50   # yeni parca arama yaricapi = bu x kayma esigi (posta cevresi)
MOTOR_STATION_FORGET = 30.0  # posta konumunda bu kadar sure motor yoksa konumu unut

IGNORE_CLASSES = {"Berkant_ibisoglu", "motor", "Muharrem_cilasin",
                  "musur_takma", "musur_aparat", "datamatrix_okutma"}

# ── SURESI OLCULEN AMA QA LISTESINDE OLMAYAN ISLEMLER ────────────────────
# Bu siniflar kontrol listesinde adim olarak GORUNMEZ (GECTI/EKSIK'i
# etkilemez) ama gorulduklerinde ISLEM (aktif calisma) sayilir: cevrim
# suresi analizine girer ve adim bazli sure raporunda ayri satir olur.
# Yeni bir islem eklemek icin sinif adini bu kumeye yazman yeterli.
IS_EXTRA_CLASSES = {"musur_takma"}

# Suresi olculen tum islem siniflari (QA adimlari + yukaridakiler)
IS_CLASSES = [c for c in QA_STEP_CLASSES if c not in SANAL_ADIMLAR] + [c for c in sorted(IS_EXTRA_CLASSES)
                                      if c not in QA_STEP_CLASSES]
IS_WORK_SET = set(IS_CLASSES)

# ── KATMA DEGER (VA/NVA) — SADECE RAPORDA, OPSIYONEL ─────────────────────
# Canli panellerde (Tk + web) VA/NVA HIC gosterilmez. Yalnizca HTML raporunda
# ve yalnizca VA_RAPOR acikken bir VA/NVA ayrimi gosterilir.
# VA = motora parca takan islemler; geri kalan izlenen islemler NVA'dir.
VA_RAPOR      = True          # False yapilirsa raporda da VA/NVA hic cikmaz
VA_OPERATIONS = {"musur_takma", "musur_skma", "pim cakma"}
NNVA_OPERATIONS = [c for c in IS_CLASSES if c not in VA_OPERATIONS]

# Sinif bazinda confidence esigi. Burada olmayan sinif genel "Islem conf"
# slider degerini kullanir. Dusuk recall'li siniflara dusuk esik ver.
# NOT: kam_mili koyma / musur_skma / motor icin degerler ARAYUZDEKI kendi
# sliderlarindan gelir, buradakiler sadece baslangic degeridir.
PER_CLASS_CONF = {
    "kep_sokme": 0.35,
    "pim cakma": 0.40,       # dusuk recall — daha kolay yakalansin
    "kam_mili koyma": 0.25,   # v8: canli kamerada kesik tespit icin 0.30 -> 0.25
    "musur_skma": 0.30,      # sahada ayarlandi (0.45 -> 0.30)
    "silikon_silme": 0.25,   # v8: kisa/aralikli goruluyor; erken yanlislari sira sarti engeller
    MOTOR_CLASS: MOTOR_CONF,
    KEP_SINIF: KEP_CONF,
    MUSUR_NESNE: MUSUR_NESNE_CONF,
    KAM_NESNE: KAM_NESNE_CONF,
}

# Adim onay suresi (dwell): sinif bu kadar sn (kisa kesintiler tolere edilerek)
# gorulmeden QA'da isaretlenmez. 0 = aninda isaretle.
# kam mili koyma ve musur sikma icin arayuzden ayarlanir.
STEP_DWELL = {
    # v8: kam mili koyma en az 0.5 sn gorulmeli (kopukluk toleransi asagida
    # STEP_GAP_TOL_ADIM'da 1.5 sn). Kalibrasyonda gercek koyma 2-9 sn, anlik
    # yanlis etiketler <= 0.4 sn. Canli kameradaki kesik tespit simulasyonunda
    # (%40 kayip) eski 1.0 sn / 0.4 sn ayari 6 koymadan ~0.2'sini, bu ayar
    # ~5.4'unu onayladi; ikisinde de yanlis onay 0.
    "kam_mili koyma": 0.5,
    "musur_skma": 0.0,
    "silikon_silme": 0.25,   # sira serbest: tek karelik yanlis tespit sayilmasin
}
STEP_GAP_TOL = 0.40   # bu kadar sn'lik kopukluk ayni gorulme sayilir
# Adima ozel kopukluk toleransi: canli kamerada etiket gidip gelince onay
# sayaci sifirlanmasin (kare atlama + agir model -> islenen kareler seyrek).
STEP_GAP_TOL_ADIM = {"kam_mili koyma": 1.5}

# ── ADIM SIRASI / KILITLER ───────────────────────────────────────────────
# STEP_REQUIRES: bir adim, gerekli adimlar bitmeden ISARETLENMEZ.
#   Model o etiketi gorse bile yok sayilir (yanlis etiket yaniltmaz).
# STEP_ZONE_GATE: adim ayrica bir BOLGE (ROI) tetiklenmeden isaretlenmez.
#   Bolge artik adimi dogrudan isaretlemez, sadece "kapiyi acar";
#   isaretlemek icin ondan SONRA model etiketi gorulmelidir.
# Gerekli adim tamamlandiginda kapi bolgesi otomatik sifirlanir; yani
# spatulanin musur sikmadan SONRA alinmasi gerekir.
STEP_ZONE_GATE = {
    # v8: "Spatula alma" bolgesi ESKI kamera acisinin koordinatlariydi; yeni
    # acida parca tepsisinin ustune dusuyor ve kapi hic acilmiyordu ->
    # silikon silme hic sayilmiyordu. Kapi kaldirildi, silikon yalnizca model
    # etiketiyle (+ asagidaki sira sarti) isaretlenir. Yeni acida spatula yeri
    # belirlenirse ROI_ZONES'a eklenip burada yeniden baglanabilir.
}

# STEP_REQUIRES_ANY: listedeki adimlardan EN AZ BIRI bitmeden isaretlenmez.
# Silikon: musur sikma VEYA musur takili (biri kacsa da silikon kilitli kalmaz).
# ERTELENEBILIR: sirasi gelmeden (yalnizca STEP_REQUIRES yuzunden) gorulen adim
# HATIRLANIR ve sart ayni cevrimde saglaninca sayilir. Canli kamerada kep
# sayimi birkac sn gec tamamlaninca o arada yapilan pim/musur sikma kayboluyordu.
# Kalibrasyonda bu adimlarin kep sokmeden ONCE hic tespiti yok (yanlis onay
# riski yok). Silikon bilerek disarida: musurden once yanlis tespitleri var.
ERTELENEBILIR = {"pim cakma", "musur_skma"}

STEP_REQUIRES_ANY = {
    # v8: silikon silme SIRADAN BAGIMSIZ (pim ve musurden once ya da sonra
    # yapilabilir). Yanlis onaya karsi sure sarti var: ana model etiketi
    # STEP_DWELL ile 0.25 sn, spatula SILIKON_MIN_SURE ile 0.5 sn.
}

STEP_REQUIRES = {
    # v8: musur_skma kep_sokme'ye BAGLI DEGIL — kep sayaci 6/6'ya ulasamazsa
    # musur sikma da eksik gorunuyordu (zincirleme hata). Musur blokta,
    # keplerden bagimsiz; yanlis tespiti konum kontrolu (motor alt govdesi) eler.
    "pim cakma":["kep_sokme"],

}

# Kapi acildiktan sonra model etiketi hic gelmezse, bu kadar sn sonra
# adimi yine de isaretle. 0 = KAPALI (varsayilan: sadece etiketle isaretlenir).
GATE_FALLBACK_SECS = 0.0

# CEVRIM ONCESI YAPILABILEN ADIMLAR
# Bu adimlar motor daha postaya yerlesmeden (motor gelirken) yapilabilir.
# Cevrim baslayinca (motor yerlesince) liste sifirlanir ama bu adimlar
# PRE_CARRY_SECS icinde yapilmissa YENI cevrime tasinir, silinmez.
PRE_CYCLE_STEPS = {"kam_mili koyma", "kep_sokme"}
PRE_CARRY_SECS  = 30.0
# Cevrim kapandiktan hemen sonra biten parcanin etiketi bir sure daha
# ekranda kalabilir; bu sure icinde gorulen adimlar YENI parcaya yazilmaz.
PRE_AFTER_CLOSE_IGNORE = 1.50

PRETTY = {
    "datamatrix_okutma": "Datamatrix okutma", "kam_mili koyma": "Kam mili koyma",
    "kep_sokme": "Kep sokme", "musur_aparat": "Musur aparat",
    "musur_skma": "Musur Sıkma", MUSUR_TAKILI_ADIM: "Müşür takılı", "musur_takma": "Musur takma",
    "pim cakma": "Pim cakma", "silikon_silme": "Silikon silme",
    "motor": "Motor",
}

# ── ROI BOLGELERI ────────────────────────────────────────────────────────
# rect = (x1, y1, x2, y2) — 0..1 arasi normalize (cozunurlukten bagimsiz)
# El (bilek veya parmak ucu) bu bolgede 'dwell_secs' kadar kalirsa
# bagli QA adimi otomatik OK isaretlenir. Arayuzden "ROI duzenle" ile
# fareyle yeniden cizebilirsin.
ROI_ZONES = [
    # v8: "Spatula alma" bolgesi kaldirildi (eski kamera acisina gore cizilmisti,
    # bkz. STEP_ZONE_GATE). Yeni acida gerekirse ayni yapida yeniden eklenebilir:
    # {"name": "Spatula alma", "rect": (x1, y1, x2, y2), "qa_step": "silikon_silme",
    #  "action": "gate", "dwell_secs": 0.2, "mode": "hand", "motion_k": 2.0, "motion_min": 12.0},
    # ── Kep Support Bolgesi (sol alt, sari/beyaz support) ─────────────────
    # Operatorun sokulen kepi biraktigi bolge. Bu bolge sadece IZLEME icin
    # kullanilir — kep sayacini dogrudan isaretlemez; KepSokmeTracker burayi
    # referans alarak kepin birakildigi anı dogrular.
    # "qa_step" None: bu bolge hicbir QA adimini tek basina isaretlemez.
    {
        "name": "Kep support",
        "rect": KEP_SUPPORT_ROI,   # (0.00, 0.72, 0.22, 1.00) — kamera acisina gore ayarla
        "qa_step": None,
        "action": "mark",           # "mark" ama qa_step=None oldugu icin etki yok
        "dwell_secs": 0.3,
        "mode": "hand",
        "motion_k": 2.0,
        "motion_min": 10.0,
    },
]

# ── YUZ BULANIKLASTIRMA (KVKK) ───────────────────────────────────────────
# Yuzler ekranda ve kaydedilen goruntulerde mozaiklenir. Bolge, iskeletin
# bas noktalarindan (burun, goz, kulak) hesaplanir; omuz genisligi olcek
# olarak kullanilir. Tespit modelleri ORIJINAL kareyi gorur, mozaik yalnizca
# gosterilen/kaydedilen goruntuye uygulanir.
YUZ_BULANIK    = True     # varsayilan acik
YUZ_MOZAIK     = 14       # mozaik kabalig (buyudukce daha az taninir)
YUZ_PAY        = 1.0      # bas bolgesi carpani (buyutmek icin >1 yap)
YUZ_RX         = 0.30     # yatay yaricap = omuz genisligi x bu
YUZ_RY         = 0.38     # dikey yaricap  (bas ovaldir, biraz uzun)
YUZ_ASAGI      = 0.06     # merkez goz hizasindan agiza dogru kaydirma
YUZ_HATIRLA    = 1.0      # bas kaybolursa bu kadar sn daha bulaniklastir
YUZ_MIN_R      = 18       # en kucuk yaricap (piksel)
# v8: KISININ TAMAMI bulaniklastirilir (yalnizca yuz degil). Bulaniklik EN ALT
# katmana uygulanir: kutular, etiketler ve iskelet bulanik goruntunun USTUNE
# cizilir. Tespit modelleri yine ORIJINAL kareyi gorur (bellekte, diske yazilmaz).
# False yaparsan eski davranis: yalnizca yuz mozaigi (yine en alt katmanda).
KISI_BULANIK   = True
KISI_PAY       = 0.06     # kisi kutusu her yandan bu oranda genisletilir
KISI_KABALIK   = 1.6      # kabalik carpani (kisi bolgesi yuze gore buyuk)
# Yalnizca INSAN VUCUDU: kisi segmentasyon modeli (yolov8n-seg.pt) klasordeyse
# kutu yerine vucudun silueti bulaniklastirilir; kollarin arasindaki motor net
# kalir. Model yoksa ya da kisiyi bulamazsa (poz modeli bulduysa) kutu bulanigi.
SEG_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-seg.pt")
KISI_SEG_CONF  = 0.25
KISI_SEG_IMGSZ = 640
KISI_GENISLET  = 0.012    # siluet her yandan kare genisliginin bu orani kadar buyutulur
# Yanlis siluet (or. motorun 'insan' sanilmasi) elemesi: siluet, poz modelinin
# buldugu bir kisinin kutusuyla en az KISI_ESLES orani kadar ortusmeli ve o
# kutunun (KISI_KIRP payiyla) disina tasamaz. Poz kimseyi bulmadiysa yalnizca
# KISI_SEG_TEK_CONF ustundeki siluetler kullanilir.
KISI_ESLES     = 0.30
KISI_KIRP      = 0.15
KISI_SEG_TEK_CONF = 0.50
# Iskelet koridoru: segmentasyon motoru 'bacak' sanip siluete katabiliyor.
# Siluetin yalnizca iskeletten cizilen vucut koridoru icindeki kismi kullanilir.
KISI_KOL_KALINLIK = 0.55  # kol/bacak koridoru kalinligi = omuz genisligi x bu
KISI_EL_UZAT      = 0.60  # el, bilekten onkol boyunun bu orani kadar ileri uzatilir

POSE_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]
L_WRIST, R_WRIST = 9, 10
L_ELBOW, R_ELBOW = 7, 8
USE_ELBOW_FALLBACK = True   # bilek gorunmezse dirsegi de bolge noktasi say
ZONE_WARMUP_SECS = 2.0      # acilis/sifirlama sonrasi bu sure tetikleme yok


def pretty(name): return PRETTY.get(name, name)


def _motor_dist(a, b):
    """Motor capasi icin mesafe: MOTOR_SADECE_YATAY ise yalnizca x farki."""
    if MOTOR_SADECE_YATAY:
        return abs(a[0] - b[0])
    return _dist(a, b)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class StepConfirm:
    """Adim onay suresi (dwell). Bir sinif kisa kesintilerle de olsa 'dwell'
    saniye boyunca gorulmeden QA'da isaretlenmez — yanlis pozitifleri keser."""
    def __init__(self, gap_tol=STEP_GAP_TOL, gap_tol_adim=None):
        self.gap_tol = gap_tol
        self.gap_tol_adim = dict(STEP_GAP_TOL_ADIM if gap_tol_adim is None else gap_tol_adim)
        self.first = {}
        self.last = {}

    def _tol(self, name):
        return self.gap_tol_adim.get(name, self.gap_tol)

    def feed(self, now, name):
        last = self.last.get(name)
        if last is None or (now - last) > self._tol(name):
            self.first[name] = now
        self.last[name] = now

    def _alive(self, now, name):
        l = self.last.get(name)
        return l is not None and (now - l) <= self._tol(name)

    def ready(self, now, name, dwell):
        if dwell <= 0.0:
            return True
        if not self._alive(now, name):
            return False
        return (now - self.first.get(name, now)) >= dwell

    def progress(self, now, name, dwell):
        if dwell <= 0.0 or not self._alive(now, name):
            return 0.0
        return min(1.0, (now - self.first.get(name, now)) / max(1e-6, dwell))

    def reset(self):
        self.first.clear()
        self.last.clear()


class MotorCycleMonitor:
    """Motorun postadaki konumunu izler — OKLUZYONA DAYANIKLI.

    Akis:
      MOTOR YOK   -> motor gorunur              -> YERLESIYOR
      YERLESIYOR  -> MOTOR_SETTLE_SECS sabit    -> POSTADA (capa kilitlendi, 'arm')
      POSTADA     -> motor kaybolursa           -> POSTADA / GORUNMUYOR  (KAPANMAZ!)
      POSTADA     -> GORULEREK capadan uzaklasir-> CIKTI ('close')
      CIKTI       -> motor MOTOR_UNDO_SECS icinde ayni yere donerse -> 'resume'
                     (yanlis kapanma geri alinir, QA silinmez)

    Onemli: motor kutusu kaybolunca hicbir sey sifirlanmaz; son bilinen konum
    saklanir, cevrim aynen devam eder. Cikis icin motorun BASKA BIR YERDE
    gorulmesi sarttir (hem sure hem tespit sayisi kosulu).
    """
    STATE_WAIT     = "MOTOR YOK"
    STATE_SETTLING = "YERLESIYOR"
    STATE_RUN      = "POSTADA"
    STATE_HIDDEN   = "GORUNMUYOR"
    STATE_OUT      = "CIKTI"

    def __init__(self):
        self.reset()

    def reset(self):
        self.smooth = None          # takip edilen (yumusatilmis) konum
        self.anchor = None          # posta konumu (capa)
        self.prev_anchor = None     # son kapanan cevrimin capasi = POSTA konumu
        self.closed_at = None
        self.station_miss_since = None
        self.empty_gap = 0.0        # kapanistan beri posta ne kadar bos kaldi
        self.settle_ref = None
        self.settle_since = None
        self.move_since = None
        self.move_hits = 0
        self.armed = False
        self.present = False
        self.displacement = 0.0
        self.last_seen = None
        self.lost_for = 0.0
        self.state = self.STATE_WAIT

    # ── ayni motoru takip et: capaya/son konuma en yakin kutuyu sec ──────
    def _pick(self, now, cands, move_thres):
        """cands: [(alan, (cx, cy)), ...] -> takip edilen motorun merkezi"""
        if not cands:
            return None

        # CEVRIM YOKKEN: yeni parca posta konumunun cevresinde aranir.
        # Boylece postadan CIKAN motor takibi ele gecirip yanlis yerde
        # cevrim baslatamaz.
        if not self.armed and self.prev_anchor is not None:
            r = max(move_thres * MOTOR_STATION_R_K, 0.05)
            near = [c for c in cands if _motor_dist(c[1], self.prev_anchor) <= r]
            if near:
                if self.station_miss_since is not None:
                    # posta ne kadar bos kaldi? (parca gercekten gitti mi?)
                    self.empty_gap = max(self.empty_gap, now - self.station_miss_since)
                self.station_miss_since = None
                return min(near, key=lambda c: _motor_dist(c[1], self.prev_anchor))[1]
            # postada motor yok: cikan motoru takip etme, bekle
            if self.station_miss_since is None:
                self.station_miss_since = now
            else:
                self.empty_gap = max(self.empty_gap, now - self.station_miss_since)
            if (self.station_miss_since is not None
                    and (now - self.station_miss_since) >= MOTOR_STATION_FORGET):
                # posta konumu artik gecerli degil (kamera/tezgah degismis olabilir)
                self.prev_anchor = None
                self.station_miss_since = None
                print("[MOTOR] posta konumu unutuldu — yeniden ogrenilecek")
            return None

        # oncelik: takip edilen konum -> capa
        ref = self.smooth or self.anchor
        if ref is None:
            # ilk kez: en buyuk kutu (postadaki asil motor)
            return max(cands, key=lambda c: c[0])[1]
        # sonrasi: referansa en yakin kutu (tekil nesne takibi)
        return min(cands, key=lambda c: _motor_dist(c[1], ref))[1]

    def update(self, now, cands, move_thres=MOTOR_MOVE_THRES,
               hold_secs=MOTOR_HOLD_SECS, absent_closes=MOTOR_ABSENT_CLOSES,
               lost_secs=MOTOR_LOST_SECS):
        """cands: bu karedeki motor kutulari [(alan,(cx,cy)),...] (bos olabilir)
        Doner: 'arm' | 'resume' | 'close' | None"""
        center = self._pick(now, cands, move_thres)

        # ── motor gorunmuyor: DONDUR, sakin kapatma ─────────────────────
        if center is None:
            self.present = False
            self.lost_for = 0.0 if self.last_seen is None else (now - self.last_seen)
            # kayma birikimini dondur (yarim kalmis kayma kapatmasin)
            self.move_since = None
            self.move_hits = 0
            if self.armed:
                # okluzyon: cevrim aynen devam eder
                self.state = self.STATE_RUN if self.lost_for <= hold_secs else self.STATE_HIDDEN
                if absent_closes and self.last_seen is not None and self.lost_for >= lost_secs:
                    self._do_close(now)
                    return "close"
            else:
                self.settle_ref = None
                self.settle_since = None
                self.state = self.STATE_WAIT
            return None

        # ── motor gorunuyor ─────────────────────────────────────────────
        gap = None if self.last_seen is None else (now - self.last_seen)
        self.present = True
        self.last_seen = now
        self.lost_for = 0.0
        if self.smooth is None or (gap is not None and gap > MOTOR_SNAP_GAP):
            self.smooth = center          # uzun kopukluk sonrasi konumu sifirdan al
        else:
            self.smooth = ((1 - MOTOR_SMOOTH) * self.smooth[0] + MOTOR_SMOOTH * center[0],
                           (1 - MOTOR_SMOOTH) * self.smooth[1] + MOTOR_SMOOTH * center[1])

        # ── capa yoksa: sabitlenmesini bekle ────────────────────────────
        if not self.armed:
            self.state = self.STATE_SETTLING
            if self.settle_ref is None or _motor_dist(self.smooth, self.settle_ref) > MOTOR_SETTLE_TOL:
                self.settle_ref = self.smooth
                self.settle_since = now
            elif (now - self.settle_since) >= MOTOR_SETTLE_SECS:
                back = (self.prev_anchor is not None and self.closed_at is not None
                        and (now - self.closed_at) <= MOTOR_UNDO_SECS
                        and self.empty_gap <= MOTOR_UNDO_EMPTY_MAX
                        and _motor_dist(self.smooth, self.prev_anchor) < move_thres)
                self.anchor = self.prev_anchor if back else self.smooth
                self.armed = True
                self.move_since = None
                self.move_hits = 0
                self.displacement = 0.0
                self.state = self.STATE_RUN
                self.prev_anchor = None
                self.closed_at = None
                self.station_miss_since = None
                return "resume" if back else "arm"
            return None

        # ── capa var: kayma olcumu (sadece GORULEN karelerde) ───────────
        self.state = self.STATE_RUN
        self.displacement = _motor_dist(self.smooth, self.anchor)
        if self.displacement >= move_thres:
            if self.move_since is None:
                self.move_since = now
                self.move_hits = 1
            else:
                self.move_hits += 1
                if ((now - self.move_since) >= MOTOR_MOVE_DEBOUNCE
                        and self.move_hits >= MOTOR_MOVE_MIN_HITS):
                    self._do_close(now)
                    return "close"
        else:
            self.move_since = None
            self.move_hits = 0
        return None

    def _do_close(self, now):
        self.prev_anchor = self.anchor
        self.closed_at = now
        self.armed = False
        self.anchor = None
        self.station_miss_since = None
        self.empty_gap = 0.0
        # takip zincirini kes: yeni parca posta konumunun yakininda aranir
        self.smooth = None
        self.settle_ref = None
        self.settle_since = None
        self.move_since = None
        self.move_hits = 0
        self.state = self.STATE_OUT


# ─────────────────────────────────────────────────────────────────────────────
class KepSokmeTracker:
    """Kep sokme kontrolu — SAYIM TABANLI (v8, op230 modeli 'kep' sinifi).

    Kalibrasyon videosunda her cevrim ayni: motor kapakta 6 keple gelir,
    operator kepleri tek tek sokup palete bagli sari/beyaz support'a dizer,
    motor support ile birlikte postadan cikar.

    * QA KARARI: support'ta KEP_COUNT kep KEP_STABIL_SECS boyunca (kayan
      medyan) sayilirsa kep_sokme TAMAM.  Tek bir etiket ya da anlik titreme
      karari degistirmez.
    * HANGI KEP: motor kapagindaki her kep tespiti KEP_HARITA'daki en yakin
      yuvaya atanir.  Bir yuva bu cevrimde gorulup KEP_YOK_SECS boyunca bir
      daha gorulmezse "sokuldu" sayilir.  Cevrim kapanirken hala motorda
      gorulen yuvalar eksik_kepler() ile raporlanir.
    * Motor kutusu operator egilince ust/alt kenardan kirpilabiliyor; bu
      yuzden yuvalar ve support, yuksekligi tam olan son kutuya (ref_box)
      gore hesaplanir.
    """

    def __init__(self, motor_mon, n=KEP_COUNT):
        self.n = n
        self.motor_mon = motor_mon
        self.reset()

    def reset(self):
        self.done        = [False] * self.n      # yuva sokuldu mu (gosterim + rapor)
        self.done_at     = [None]  * self.n
        self.son_gorulme = [None]  * self.n      # yuva motorda en son ne zaman goruldu
        self.motor_box   = None                  # son tam boylu motor kutusu (ref)
        self._ref_h      = 0.0
        self._sup_hist   = collections.deque()   # (zaman, support_sayisi)
        self._mot_hist   = collections.deque()   # (zaman, motordaki_sayi)
        self.sup_merkez  = None                  # support merkezi (keplerden ogrenilen)
        self.son_konum   = [None] * self.n       # her yuvanin en son goruldugu ekran konumu
        self.support_n   = 0                     # kararli (medyan) support sayisi
        self.motor_n     = 0                     # kararli motordaki kep sayisi
        self.tamam_at    = None

    # ── kutu: yuksekligi tam olan son kutu referans alinir ─────────────
    def _ref_guncelle(self, box):
        if box is None:
            return
        h = box[3] - box[1]
        if self.motor_box is None or h >= self._ref_h * KEP_REF_MIN_H:
            self.motor_box = box
            self._ref_h = max(h, self._ref_h * 0.98)   # yavas unut (motor degisirse)
        else:
            # Guncellenmese de referans yuksekligi yavasca azalir: tek karelik
            # anormal buyuk bir kutu referansi kalici olarak kilitlemesin
            # (~20 karede kendini toparlar).
            self._ref_h *= 0.995

    def support_kutusu(self):
        """Ogrenilen support merkezi varsa onun cevresi, yoksa motora goreli tahmin."""
        if self.sup_merkez is not None:
            cx, cy = self.sup_merkez; hx, hy = KEP_SUPPORT_YARI
            return (cx - hx, cy - hy, cx + hx, cy + hy)
        return support_rect(self.motor_box)

    def en_yakin_kep(self, rx, ry):
        best, bd = None, KEP_MATCH_R
        for i, (kx, ky) in enumerate(KEP_HARITA[:self.n]):
            d = math.hypot(rx - kx, ry - ky)
            if d <= bd:
                best, bd = i, d
        return best

    @staticmethod
    def _medyan(hist):
        v = sorted(x for _, x in hist)
        return v[len(v) // 2] if v else 0

    def feed(self, now, kep_boxes, motor_box=None, support_active=False):
        """kep_boxes: bu karedeki 'kep' tespitlerinin normalize merkezleri.
        Doner: bu cagrida 'sokuldu' sayilan yuva indeksleri."""
        self._ref_guncelle(motor_box)
        box = self.motor_box
        if box is None:
            return []
        mx1, my1, mx2, my2 = box
        mw, mh = max(1e-6, mx2 - mx1), max(1e-6, my2 - my1)
        kapak, disari = [], []
        for cx, cy in kep_boxes:
            rx, ry = (cx - mx1) / mw, (cy - my1) / mh
            if -KEP_BOX_PAD <= rx <= 1 + KEP_BOX_PAD and ry <= KEP_KAPAK_MAX_RY:
                kapak.append((cx, cy, rx, ry))
            elif -1.2 <= rx <= 0.3 and 0.3 <= ry <= 1.4:   # paletin support tarafi
                disari.append((cx, cy))
        # support merkezini keplerin kendisinden ogren (motor donse de dogru kalir)
        if len(disari) >= 2:
            mcx = sorted(c[0] for c in disari)[len(disari) // 2]
            mcy = sorted(c[1] for c in disari)[len(disari) // 2]
            if self.sup_merkez is None:
                self.sup_merkez = (mcx, mcy)
            else:
                a = KEP_SUPPORT_OGREN
                self.sup_merkez = ((1 - a) * self.sup_merkez[0] + a * mcx,
                                   (1 - a) * self.sup_merkez[1] + a * mcy)
        sx1, sy1, sx2, sy2 = self.support_kutusu()
        n_sup = sum(1 for cx, cy in disari if sx1 <= cx <= sx2 and sy1 <= cy <= sy2)
        n_mot = len(kapak)
        for cx, cy, rx, ry in kapak:
            slot = self.en_yakin_kep(rx, ry)
            if slot is not None:
                self.son_gorulme[slot] = now
                self.son_konum[slot] = (cx, cy)
        for hist, val in ((self._sup_hist, n_sup), (self._mot_hist, n_mot)):
            hist.append((now, val))
            while hist and now - hist[0][0] > KEP_STABIL_SECS:
                hist.popleft()
        self.support_n = self._medyan(self._sup_hist)
        self.motor_n = self._medyan(self._mot_hist)

        # Yuva KEP_YOK_SECS gorulmezse "sokuldu" ADAYI olur; ama el kepin
        # ustunu kapatinca da gorulmez. Bu yuzden sokuldu sayilan yuva sayisi
        # support'taki kep sayisini GECEMEZ (en uzun suredir gorulmeyenler once).
        newly = []
        aday = [i for i in range(self.n) if not self.done[i] and self.son_gorulme[i] is not None
                and now - self.son_gorulme[i] >= KEP_YOK_SECS]
        izin = self.support_n - sum(self.done)
        for i in sorted(aday, key=lambda j: self.son_gorulme[j])[:max(0, izin)]:
            self.done[i] = True; self.done_at[i] = now; newly.append(i)
        if self.support_n >= self.n and self.tamam_at is None:
            self.tamam_at = now
            for i in range(self.n):          # support dolu: hepsi sokulmus demektir
                if not self.done[i]:
                    self.done[i] = True; self.done_at[i] = now
            print(f"[KEP] support'ta {self.support_n}/{self.n} kep -> kep sokme TAMAM")
        return newly

    def eksik_kepler(self, now):
        """Cevrim kapanirken hala motorda gorulen yuvalar (1'den baslayan no)."""
        return [i + 1 for i, t in enumerate(self.son_gorulme)
                if t is not None and now - t < KEP_YOK_SECS + 1.0 and self.tamam_at is None]

    @property
    def tamamlanan(self):
        return max(sum(self.done), min(self.support_n, self.n))

    @property
    def tamamlandi_mi(self):
        return self.tamam_at is not None

    def durum_metni(self):
        isaretler = "".join("v" if d else "-" for d in self.done)
        return f"support {self.support_n}/{self.n}, motorda {self.motor_n} ({isaretler})"

    def qa_gecti_mi(self):
        return self.tamamlandi_mi


class QAChecklist:
    def __init__(self, steps):
        self.steps = list(steps)
        self.done = {s: False for s in self.steps}
        self.source = {s: "" for s in self.steps}   # "model" | "bolge"

    def missing(self):
        """Eksik kalan adimlar."""
        return [s for s in self.steps if not self.done[s]]

    def mark(self, name, src="model"):
        if name not in self.done or self.done[name]:
            return
        self.done[name] = True
        self.source[name] = src

    def reset(self):
        for s in self.done:
            self.done[s] = False
            self.source[s] = ""

    @property
    def ok_count(self): return sum(1 for s in self.steps if self.done[s])
    @property
    def total(self): return len(self.steps)
    @property
    def passed(self): return self.total > 0 and self.ok_count == self.total
    def snapshot(self): return dict(self.done), dict(self.source), self.ok_count, self.total, self.passed


class ZoneMonitor:
    """El bolgeye girerse VEYA bolgede olagandisi hareket olursa tetiklenir.
    Hareket esigi sahnenin kendi gurultusune gore otomatik uyarlanir."""
    def __init__(self, cfg):
        self.name = cfg["name"]
        self.rect = tuple(cfg["rect"])
        self.qa_step = cfg.get("qa_step")
        self.action = cfg.get("action", "mark")   # "mark" | "gate"
        self.dwell_secs = float(cfg.get("dwell_secs", 0.5))
        self.mode = cfg.get("mode", "hand")
        self.motion_k = float(cfg.get("motion_k", 3.0))       # taban gurultunun kac kati
        self.motion_min = float(cfg.get("motion_min", 12.0))  # mutlak alt sinir
        self.inside_since = None
        self.hand_inside = False
        self.motion_active = False
        self.motion_score = 0.0
        self.baseline = 0.0          # sahnenin normal gurultu seviyesi
        self._prev_roi = None
        self.progress = 0.0
        self.triggered = False
        self.last_trigger_t = None
        self.warmup_until = time.time() + ZONE_WARMUP_SECS

    @property
    def motion_thres(self):
        return max(self.motion_min, self.baseline * self.motion_k)

    def px(self, w, h):
        x1, y1, x2, y2 = self.rect
        return int(x1*w), int(y1*h), int(x2*w), int(y2*h)

    def _motion(self, frame):
        """Bolgedeki degisimi olcer ve taban gurultuyu ogrenir."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.px(w, h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return 0.0
        roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (64, 64))
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        if self._prev_roi is None:
            self._prev_roi = roi
            return 0.0
        score = float(cv2.absdiff(roi, self._prev_roi).mean())
        self._prev_roi = roi
        self.motion_score = 0.6 * self.motion_score + 0.4 * score
        # taban gurultu: yalnizca sakin anlarda yavasca ogren
        if self.motion_score < self.motion_thres:
            self.baseline = 0.98 * self.baseline + 0.02 * self.motion_score
        elif self.baseline == 0.0:
            self.baseline = self.motion_score * 0.5
        return self.motion_score

    def update(self, now, points, frame):
        """points: bilek / dirsek / parmak ucu (frame piksel koordinati)"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.px(w, h)
        self.hand_inside = any(x1 <= px <= x2 and y1 <= py <= y2 for px, py in points)
        score = self._motion(frame)
        self.motion_active = score >= self.motion_thres

        if self.mode == "hand":
            active = self.hand_inside
        elif self.mode == "motion":
            active = self.motion_active
        else:
            active = self.hand_inside or self.motion_active

        # isinma suresinde tetikleme yok (taban gurultu otursun)
        if now < self.warmup_until:
            self.inside_since = None
            self.progress = 0.0
            return False

        fired = False
        if active:
            if self.inside_since is None:
                self.inside_since = now
            self.progress = min(1.0, (now - self.inside_since) / max(1e-6, self.dwell_secs))
            if self.progress >= 1.0 and not self.triggered:
                self.triggered = True
                self.last_trigger_t = now
                fired = True
        else:
            self.inside_since = None
            self.progress = 0.0
        return fired

    def reset(self, warmup=None):
        self.inside_since = None
        self.hand_inside = False
        self.motion_active = False
        self.motion_score = 0.0
        self._prev_roi = None
        self.progress = 0.0
        self.triggered = False
        self.last_trigger_t = None
        self.warmup_until = time.time() + (ZONE_WARMUP_SECS if warmup is None else warmup)


def bas_bolgeleri(keypoints, w, h, kpt_thres=0.15):
    """Iskelet noktalarindan bas bolgelerini (cx, cy, r) olarak dondurur."""
    bolgeler = []
    if keypoints is None:
        return bolgeler
    try:
        xy = keypoints.xy.cpu().numpy()
        conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None
    except Exception:
        return bolgeler
    for pi in range(len(xy)):
        pts = xy[pi]
        c = conf[pi] if conf is not None else np.ones(len(pts))
        bas = [(float(pts[i][0]), float(pts[i][1])) for i in range(min(5, len(pts)))
               if c[i] > kpt_thres and pts[i][0] > 0 and pts[i][1] > 0]
        if not bas:
            continue
        cx = sum(p[0] for p in bas) / len(bas)
        cy = sum(p[1] for p in bas) / len(bas)
        # olcek: omuz genisligi varsa ondan, yoksa bas noktalarinin yayilimindan
        d = 0.0
        if len(pts) > 6 and c[5] > kpt_thres and c[6] > kpt_thres:
            d = math.hypot(pts[5][0] - pts[6][0], pts[5][1] - pts[6][1])  # omuz genisligi
        if d <= 0 and len(bas) > 1:
            # omuz yoksa: bas noktalarinin yayilimindan tahmin (kulaklar arasi ~ d/3)
            d = 3.0 * max(math.hypot(a[0]-b[0], a[1]-b[1]) for a in bas for b in bas)
        if d <= 0:
            d = 4 * YUZ_MIN_R
        rx = max(YUZ_MIN_R, d * YUZ_RX * YUZ_PAY)
        ry = max(YUZ_MIN_R, d * YUZ_RY * YUZ_PAY)
        cy += d * YUZ_ASAGI                      # merkez: goz hizasindan biraz asagi
        if -rx <= cx <= w + rx and -ry <= cy <= h + ry:
            bolgeler.append((cx, cy, rx, ry))
    return bolgeler


def yuz_mozaik(img, bolgeler, kabalik=None):
    """Verilen bas bolgelerini eliptik maskeyle mozaikler."""
    if img is None or not bolgeler:
        return img
    kabalik = max(4, int(kabalik or YUZ_MOZAIK))
    h, w = img.shape[:2]
    for b in bolgeler:
        cx, cy = b[0], b[1]
        rx, ry = (b[2], b[3]) if len(b) > 3 else (b[2], b[2])
        x1, y1 = int(max(0, cx - rx)), int(max(0, cy - ry))
        x2, y2 = int(min(w, cx + rx)), int(min(h, cy + ry))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        roi = img[y1:y2, x1:x2]
        kh = max(1, (y2 - y1) // kabalik)
        kw = max(1, (x2 - x1) // kabalik)
        kucuk = cv2.resize(roi, (kw, kh), interpolation=cv2.INTER_LINEAR)
        mozaik = cv2.resize(kucuk, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        maske = np.zeros((y2 - y1, x2 - x1), np.uint8)
        cv2.ellipse(maske, ((x2-x1)//2, (y2-y1)//2),
                    ((x2-x1)//2, (y2-y1)//2), 0, 0, 360, 255, -1)
        maske = cv2.GaussianBlur(maske, (9, 9), 0)
        m3 = cv2.merge([maske, maske, maske]).astype(np.float32) / 255.0
        img[y1:y2, x1:x2] = (mozaik * m3 + roi * (1 - m3)).astype(np.uint8)
    return img


class IpOkuyucu:
    """IP kamerayi AYRI bir is parcaciginda surekli okur, yalnizca SON kareyi tutar.
    Cikarim yavasladiginda akis zamaninda okunmazsa kamera paket atar ve
    goruntu gri/bulanik parcalara doner; surekli okuma bunu onler.
    cv2.VideoCapture yerine kullanilir (read/get/set/isOpened/release)."""
    def __init__(self, cap, bekle=2.0):
        self.cap, self.bekle = cap, bekle
        self.kilit = threading.Lock()
        self.kare, self.no, self.son_no, self.saglam = None, 0, -1, True
        self.dur = False
        self.atlanan = 0                      # cikarimin okumadigi kare sayisi (bilgi)
        self.t = threading.Thread(target=self._dongu, daemon=True, name="ip-okuyucu")
        self.t.start()

    def _dongu(self):
        while not self.dur:
            ok, f = self.cap.read()
            if not ok:
                self.saglam = False
                time.sleep(0.05)
                continue
            with self.kilit:
                if self.no != self.son_no and self.kare is not None:
                    self.atlanan += 1
                self.kare, self.saglam = f, True
                self.no += 1

    def read(self):
        t0 = time.time()
        while time.time() - t0 < self.bekle:
            with self.kilit:
                if self.kare is not None and self.no != self.son_no:
                    self.son_no = self.no
                    return True, self.kare
            if not self.saglam and time.time() - t0 > 0.5:
                break
            time.sleep(0.003)
        return False, None

    def grab(self):
        return True

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, p):
        return self.cap.get(p)

    def set(self, p, v):
        return self.cap.set(p, v)

    def release(self):
        self.dur = True
        try:
            self.t.join(timeout=2.0)
        except Exception:
            pass
        self.cap.release()


def kisi_kutulari(boxes, w, h, pay=None):
    """Poz modelinin kisi kutularini (pay ile genisletilmis) piksel olarak dondurur."""
    pay = KISI_PAY if pay is None else pay
    kut = []
    if boxes is None or len(boxes) == 0:
        return kut
    try:
        xy = boxes.xyxy.cpu().numpy()
    except Exception:
        return kut
    for x1, y1, x2, y2 in xy:
        px, py = (x2 - x1) * pay, (y2 - y1) * pay
        x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            kut.append((x1, y1, x2, y2))
    return kut


def kisi_bulanik(img, kutular, kabalik=None):
    """Kisi kutularini guclu bulanik/mozaik ile kapatir (kenarlari yumusak)."""
    if img is None or not kutular:
        return img
    blok = max(6, int((kabalik or YUZ_MOZAIK) * KISI_KABALIK))
    h, w = img.shape[:2]
    for kx1, ky1, kx2, ky2 in kutular:
        # kutunun ICI tamamen bulanik; yumusak gecis kutunun DISINDA (kenar payi)
        kenar = max(3, min(kx2 - kx1, ky2 - ky1) // 12)
        x1, y1 = max(0, kx1 - kenar), max(0, ky1 - kenar)
        x2, y2 = min(w, kx2 + kenar), min(h, ky2 + kenar)
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 4:
            continue
        roi = img[y1:y2, x1:x2]
        kucuk = cv2.resize(roi, (max(1, bw // blok), max(1, bh // blok)), interpolation=cv2.INTER_AREA)
        bulanik = cv2.resize(kucuk, (bw, bh), interpolation=cv2.INTER_LINEAR)
        maske = np.zeros((bh, bw), np.uint8)
        cv2.rectangle(maske, (kx1 - x1, ky1 - y1), (kx2 - x1, ky2 - y1), 255, -1)
        maske = cv2.GaussianBlur(maske, (0, 0), kenar / 2.0)
        cv2.rectangle(maske, (kx1 - x1, ky1 - y1), (kx2 - x1, ky2 - y1), 255, -1)   # ic tam
        m3 = (maske.astype(np.float32) / 255.0)[..., None]
        img[y1:y2, x1:x2] = (bulanik * m3 + roi * (1 - m3)).astype(np.uint8)
    return img


def kisi_koridor(keypoints, w, h, kpt_thres=0.15):
    """Iskelet noktalarindan comert bir vucut koridoru maskesi (0/255) cizer.
    Hicbir kisi icin yeterli nokta yoksa None dondurur."""
    if keypoints is None:
        return None
    try:
        xy = keypoints.xy.cpu().numpy(); cf = keypoints.conf.cpu().numpy()
    except Exception:
        return None
    m = np.zeros((h, w), np.uint8)
    ciz = False
    for P, C in zip(xy, cf):
        ok = lambda i: i < len(P) and C[i] > kpt_thres and P[i][0] > 0 and P[i][1] > 0
        pt = lambda i: (int(P[i][0]), int(P[i][1]))
        if ok(5) and ok(6):
            d = math.hypot(P[5][0] - P[6][0], P[5][1] - P[6][1])
        else:
            vis = [P[i] for i in range(len(P)) if ok(i)]
            if len(vis) < 2:
                continue
            d = 0.5 * max(math.hypot(a[0] - b[0], a[1] - b[1]) for a in vis for b in vis)
        d = max(d, 20.0)
        t = max(10, int(d * KISI_KOL_KALINLIK))
        # govde: omuzlar + kalcalar (kalca yoksa omuzdan asagi uzat)
        if ok(5) and ok(6):
            alt = [pt(12) if ok(12) else (pt(6)[0], int(pt(6)[1] + 1.3 * d)),
                   pt(11) if ok(11) else (pt(5)[0], int(pt(5)[1] + 1.3 * d))]
            cv2.fillPoly(m, [np.array([pt(5), pt(6), alt[0], alt[1]], np.int32)], 255)
            cv2.polylines(m, [np.array([pt(5), pt(6), alt[0], alt[1]], np.int32)], True, 255, t)
            ciz = True
        for a, b in ((5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6)):
            if ok(a) and ok(b):
                cv2.line(m, pt(a), pt(b), 255, t); ciz = True
        for dir_, bil in ((7, 9), (8, 10)):          # eller: bilekten ileri
            if ok(bil):
                bx, by = P[bil][0], P[bil][1]
                if ok(dir_):
                    ex, ey = bx + (bx - P[dir_][0]) * KISI_EL_UZAT, by + (by - P[dir_][1]) * KISI_EL_UZAT
                    cv2.line(m, pt(bil), (int(ex), int(ey)), 255, t)
                    cv2.circle(m, (int(ex), int(ey)), int(t * 0.8), 255, -1)
                cv2.circle(m, pt(bil), int(t * 0.9), 255, -1); ciz = True
        bas = [i for i in range(5) if ok(i)]          # bas
        if bas:
            cx = int(np.mean([P[i][0] for i in bas])); cy = int(np.mean([P[i][1] for i in bas]))
            cv2.ellipse(m, (cx, cy), (int(0.55 * d), int(0.7 * d)), 0, 0, 360, 255, -1)
            if ok(5) and ok(6):
                cv2.line(m, (cx, cy), ((pt(5)[0] + pt(6)[0]) // 2, (pt(5)[1] + pt(6)[1]) // 2), 255, t)
            ciz = True
    return m if ciz else None


def kisi_siluet_bulanik(img, maske, kabalik=None):
    """Kisi siluetini (0/255 maske) genisletip bulaniklastirir; disi net kalir."""
    if img is None or maske is None:
        return img
    h, w = img.shape[:2]
    gen = max(3, int(w * KISI_GENISLET))
    m = cv2.dilate(maske, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gen + 1, 2 * gen + 1)))
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return img
    x1, x2 = max(0, xs.min() - gen), min(w, xs.max() + gen + 1)
    y1, y2 = max(0, ys.min() - gen), min(h, ys.max() + gen + 1)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return img
    blok = max(6, int((kabalik or YUZ_MOZAIK) * KISI_KABALIK))
    roi = img[y1:y2, x1:x2]
    kucuk = cv2.resize(roi, (max(1, bw // blok), max(1, bh // blok)), interpolation=cv2.INTER_AREA)
    bulanik = cv2.resize(kucuk, (bw, bh), interpolation=cv2.INTER_LINEAR)
    mr = m[y1:y2, x1:x2]
    yumusak = np.maximum(cv2.GaussianBlur(mr, (0, 0), gen / 2.0), mr)   # ic tam, kenar yumusak
    m3 = (yumusak.astype(np.float32) / 255.0)[..., None]
    img[y1:y2, x1:x2] = (bulanik * m3 + roi * (1 - m3)).astype(np.uint8)
    return img


def draw_pose(img, keypoints, kpt_thres=0.15):
    """Iskelet cizer; bilekleri vurgular. Bolge icin el noktalarini dondurur.
    Bilek gorunmezse dirsek yedek olarak kullanilir."""
    points = []
    if keypoints is None:
        return img, points
    try:
        xy = keypoints.xy.cpu().numpy()
        conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None
    except Exception:
        return img, points

    for pi in range(len(xy)):
        pts = xy[pi]
        c = conf[pi] if conf is not None else np.ones(len(pts))
        for a, b in POSE_SKELETON:
            if c[a] > kpt_thres and c[b] > kpt_thres:
                pa, pb = pts[a], pts[b]
                if pa[0] > 0 and pa[1] > 0 and pb[0] > 0 and pb[1] > 0:
                    cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                             (0, 255, 255), 2, cv2.LINE_AA)

        wrist_ok = {L_WRIST: False, R_WRIST: False}
        for i, (x, y) in enumerate(pts):
            if c[i] <= kpt_thres or x <= 0 or y <= 0:
                continue
            if i in (L_WRIST, R_WRIST):
                wrist_ok[i] = True
                cv2.circle(img, (int(x), int(y)), 9, (255, 0, 200), 2, cv2.LINE_AA)
                cv2.circle(img, (int(x), int(y)), 4, (255, 0, 200), -1, cv2.LINE_AA)
                points.append((float(x), float(y)))
            else:
                cv2.circle(img, (int(x), int(y)), 3, (0, 140, 255), -1, cv2.LINE_AA)

        # bilek yoksa dirsegi yedek nokta olarak kullan
        if USE_ELBOW_FALLBACK:
            for elbow, wrist in ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)):
                if not wrist_ok[wrist] and c[elbow] > kpt_thres:
                    ex, ey = pts[elbow]
                    if ex > 0 and ey > 0:
                        cv2.circle(img, (int(ex), int(ey)), 7, (255, 120, 220), 1, cv2.LINE_AA)
                        points.append((float(ex), float(ey)))
    return img, points


def draw_zone(img, zone, w, h):
    """Bolge sinirini durum rengiyle cizer: gri=bos, sari=aktif, yesil=tetiklendi."""
    x1, y1, x2, y2 = zone.px(w, h)
    if zone.triggered:
        color, thick = (0, 220, 60), 3
        state = "KAPI ACIK" if zone.action == "gate" else "OK"
    elif zone.hand_inside or zone.motion_active:
        color, thick = (0, 215, 255), 3
        state = "EL" if zone.hand_inside else "HRK"
    else:
        color, thick = (170, 170, 170), 2
        state = ""

    # kose isaretli cerceve
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
    cl = max(10, int((x2 - x1) * 0.22))
    for (cx, cy, dx, dy) in ((x1,y1,1,1), (x2,y1,-1,1), (x1,y2,1,-1), (x2,y2,-1,-1)):
        cv2.line(img, (cx, cy), (cx + dx*cl, cy), color, thick + 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy*cl), color, thick + 2, cv2.LINE_AA)

    # etiket
    label = f"{zone.name} {state}".strip()
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ly = max(th + 6, y1 - 6)
    ov = img.copy()
    cv2.rectangle(ov, (x1, ly - th - 5), (x1 + tw + 8, ly + 4), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(ov, 0.55, img, 0.45, 0)
    cv2.putText(img, label, (x1 + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # bekleme (dwell) ilerleme cubugu + hareket skoru
    if not zone.triggered and zone.progress > 0:
        bw = int((x2 - x1) * zone.progress)
        cv2.rectangle(img, (x1, y2 + 3), (x1 + bw, y2 + 8), (0, 215, 255), -1)
    cv2.putText(img, f"hrk {zone.motion_score:.1f}/{zone.motion_thres:.0f}",
                (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return img


def draw_motor_gate(img, mon, move_thres, w, h):
    """Motor capa noktasini ve cikis esigini cizer (ayar yaparken cok isine yarar).
    Yesil daire = calisma alani; motor bu daireden cikinca cevrim kapanir."""
    if mon.anchor is None or not mon.armed:
        if mon.smooth is not None:
            cx, cy = int(mon.smooth[0]*w), int(mon.smooth[1]*h)
            cv2.circle(img, (cx, cy), 6, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.putText(img, "motor yerlesiyor", (cx + 10, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
        return img
    ax, ay = int(mon.anchor[0]*w), int(mon.anchor[1]*h)
    r = max(6, int(move_thres * w))
    if not mon.present:
        color = (160, 160, 160)          # okluzyon: gri, cevrim donduruldu
    elif mon.displacement >= move_thres * 0.7:
        color = (0, 165, 255)            # cikisa yakin
    else:
        color = (0, 220, 60)             # postada
    cv2.circle(img, (ax, ay), r, color, 2, cv2.LINE_AA)
    cv2.drawMarker(img, (ax, ay), color, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    if mon.smooth is not None:
        mx, my = int(mon.smooth[0]*w), int(mon.smooth[1]*h)
        cv2.line(img, (ax, ay), (mx, my), color, 2, cv2.LINE_AA)
        cv2.circle(img, (mx, my), 7, color, -1 if mon.present else 2, cv2.LINE_AA)
    txt = (f"motor kayma {mon.displacement:.3f}/{move_thres:.2f}" if mon.present
           else f"motor gorunmuyor {mon.lost_for:.1f}sn — cevrim devam")
    cv2.putText(img, txt, (ax - r, ay - r - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def draw_kep_tracker(img, kep_tracker, support_roi, w, h, musur_ok=None, musur_poz=None):
    """Kep/musur durumunu cizer. Halkalar sabit haritaya DEGIL, nesnenin
    GERCEKTEN goruldugu yere cizilir; motor palette dondurulse de kaymaz.
    - Kep yuvalari: son goruldukleri konumda (yesil = sokuldu, kirmizi = bekliyor).
      Kep sokme TAMAM olunca halkalar gizlenir, yalnizca ozet yazisi kalir.
    - Support: uzerindeki keplerden ogrenilen konum.
    - Musur: gercek tespit konumunda halka; henuz gorulmediyse yalnizca yazi.
    support_roi: eski surumle uyum icin; kullanilmaz.
    """
    n = kep_tracker.n
    box = kep_tracker.motor_box
    sr = kep_tracker.support_kutusu()
    sx1, sy1 = int(sr[0] * w), int(sr[1] * h)
    sx2, sy2 = int(sr[2] * w), int(sr[3] * h)
    s_renk = (60, 200, 60) if kep_tracker.support_n >= n else (0, 215, 255)
    cv2.rectangle(img, (sx1, sy1), (sx2, sy2), s_renk, 2, cv2.LINE_AA)
    cv2.putText(img, f"KEP SUPPORT {min(kep_tracker.support_n, n)}/{n}", (sx1 + 3, sy1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_renk, 1, cv2.LINE_AA)
    if box is None:
        return img
    mx1, my1, mx2, my2 = box
    mw, mh = mx2 - mx1, my2 - my1
    r = max(10, int(0.045 * mw * w))
    if not kep_tracker.tamamlandi_mi:
        for i, done in enumerate(kep_tracker.done):
            poz = kep_tracker.son_konum[i]
            if poz is None:
                kx, ky = KEP_HARITA[i]
                poz = (mx1 + kx * mw, my1 + ky * mh)
            cx, cy = int(poz[0] * w), int(poz[1] * h)
            renk = (60, 200, 60) if done else (60, 60, 230)
            cv2.circle(img, (cx, cy), r, renk, 3, cv2.LINE_AA)
            t = str(i + 1)
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            tx, ty = cx - tw // 2, cy - r - 5
            cv2.rectangle(img, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), renk, -1)
            cv2.putText(img, t, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
    # musur: yalnizca gercekten goruldugu yerde halka
    if musur_poz is not None:
        ux, uy = int(musur_poz[0] * w), int(musur_poz[1] * h)
        renk = (60, 200, 60) if musur_ok else (0, 140, 255)
        cv2.circle(img, (ux, uy), int(r * 1.2), renk, 3, cv2.LINE_AA)
        t = "MUSUR"
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        tx, ty = ux - tw // 2, uy - int(r * 1.2) - 6
        cv2.rectangle(img, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), renk, -1)
        cv2.putText(img, t, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
    kep_t = (f"KEP {n}/{n} TAMAM" if kep_tracker.tamamlandi_mi else
             f"KEP {min(kep_tracker.support_n, n)}/{n} sokuldu")
    mus_t = "MUSUR TAKILI" if musur_ok else "MUSUR bekleniyor"
    etk = f"{kep_t}   |   {mus_t}"
    (tw, th), _ = cv2.getTextSize(etk, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    tx, ty = int(mx1 * w), max(th + 8, int(my1 * h) - 2 * r - 22)
    cv2.rectangle(img, (tx - 4, ty - th - 6), (tx + tw + 4, ty + 4), (20, 20, 20), -1)
    cv2.putText(img, etk, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (60, 200, 60) if (kep_tracker.tamamlandi_mi and musur_ok) else (0, 215, 255),
                2, cv2.LINE_AA)
    return img


# ══ URETIM KAYDI VE ANALIZ ═══════════════════════════════════════════════
# Her cevrim (posta sifirlanmasindan digerine) bir satir olarak gunluk CSV'ye
# yazilir. Rapor butonu bu CSV'lerden saatlik + gunluk analiz uretir.
#
# Sureler:
#   bekleme_sn   : onceki parca ciktiktan yeni parca yerlesene kadar (bos)
#   posta_sn     : motor postada gecirdigi sure
#   va_sn        : posta suresi icinde bir ISLEM etiketi gorulen saniyeler
#                  (sutun adi eski dosyalarla uyum icin korundu)
#   nva_posta_sn : posta suresi icinde islem gorulmeyen (bos) saniyeler
#   cevrim_sn    : bekleme + posta  (= sifirlamadan sifirlamaya)
#   cmin         : cevrim suresi santidakika cinsinden (1 dk = 100 cmin)
LOG_DIR      = os.path.join(BASE_DIR, "kayitlar")
SEC_PER_CMIN = 0.6          # 1 dakika = 100 cmin -> 1 cmin = 0,6 sn
ACC_DT_MAX   = 2.0          # tek karede sayilacak azami sure (donma korumasi)

# ── VARDIYALAR ───────────────────────────────────────────────────────────
# (ad, baslangic, bitis) — gun icindeki dakika cinsinden. Rapor SADECE bu
# araliklardaki cevrimleri gosterir; disinda kalan saatler (mola, gece)
# grafigi bozmasin diye tamamen atilir.
SHIFTS = [("Vardiya 1", 7*60 + 30, 15*60 + 30),     # 07:30 - 15:30
          ("Vardiya 2", 15*60 + 30, 23*60 + 30)]    # 15:30 - 23:30

# Bu sureden uzun cevrimler uretim degil DURUS sayilir: ortalamalara,
# grafiklere ve hiz hesabina girmez, ayrica raporlanir.
CEVRIM_MAX_SN = 300.0       # 5 dakika

# Bir islem etiketi bu kadar sn kaybolup geri gelirse AYNI islem sayilir.
# "Iki etiket arasi" olcum bu toleransla yapilir.
SPAN_GAP = 1.0
SON_CEVRIM_N = 40           # "son cevrimler" grafiginde kac cevrim gosterilsin


def _sanit(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").lower()


STEP_COL = {s: _sanit(s) for s in IS_CLASSES}

CYCLE_FIELDS = (["tarih", "saat", "vardiya", "baslangic", "bitis", "cevrim_sn",
                 "cmin", "parca_saat", "bekleme_sn", "posta_sn", "va_sn",
                 "nva_posta_sn", "nva_toplam_sn", "va_yuzde",
                 # "iki etiket arasi" olcum: islem araliklari birlestirilerek
                 "va_span_sn", "nnva_span_sn", "nva_bas_sn", "nva_ara_sn", "nva_son_sn",
                 "sonuc", "ok_adim", "toplam_adim", "eksik"]
                + [f"sure_{c}" for c in STEP_COL.values()]
                + [f"bitis_{c}" for c in STEP_COL.values()])


def _num(v, d=2):
    """CSV'ye Excel-TR uyumlu (virgullu) sayi yaz."""
    try:
        return f"{float(v):.{d}f}".replace(".", ",")
    except Exception:
        return ""


def _f(v, default=0.0):
    """CSV'den sayi oku (virgul veya nokta)."""
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return default


def nf(v, d=1):
    """Ekran/rapor icin virgullu sayi."""
    return f"{v:.{d}f}".replace(".", ",")


# ── SURE BIRIMI ──────────────────────────────────────────────────────────
# Ekranda ve raporda sureler santidakika (cmin) olarak gosterilir.
# 1 dakika = 100 cmin  ->  1 cmin = 0,6 sn.  Saniyeye donmek icin: BIRIM = "sn"
BIRIM = "cmin"

def sb():
    """Birim etiketi."""
    return "cmin" if BIRIM == "cmin" else "sn"

def sg(sn, d=None):
    """Saniyeyi secili birimde metne cevirir (birimsiz sayi)."""
    if BIRIM == "cmin":
        return nf(sn / SEC_PER_CMIN, 0 if d is None else d)
    return nf(sn, 1 if d is None else d)

def sd(sn):
    """Saniyeyi secili birimde sayiya cevirir (grafikler icin)."""
    return sn / SEC_PER_CMIN if BIRIM == "cmin" else sn

def st_(sn, d=None):
    """'75 cmin' seklinde tam metin."""
    return f"{sg(sn, d)} {sb()}"


def vardiya_adi(dt):
    """Verilen zamanin hangi vardiyaya dustugu ('' = vardiya disi)."""
    dk = dt.hour * 60 + dt.minute
    for ad, b, s in SHIFTS:
        if b <= s:
            if b <= dk < s:
                return ad
        else:                       # gece yarisini asan vardiya
            if dk >= b or dk < s:
                return ad
    return ""


def _merge_spans(spans):
    """Tum islem araliklarini birlestirir -> [[bas, bit], ...] (ust uste binenler tek)."""
    ivs = sorted([list(iv) for lst in spans.values() for iv in lst])
    out = []
    for iv in ivs:
        if out and iv[0] <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], iv[1])
        else:
            out.append(list(iv))
    return out


def span_feed(cyc, name, now, gap=SPAN_GAP):
    """Bir islem etiketi goruldu: aralik ac veya suregeleni uzat."""
    sp = cyc["spans"].setdefault(name, [])
    if sp and (now - sp[-1][1]) <= gap:
        sp[-1][1] = now
    else:
        sp.append([now, now])


def _normalize(r):
    """Okunan satiri tamamlar: eksik sutunlari eski verilerden turetir.

    Eski surumle (span sutunlari olmadan) yazilmis kayitlar da raporda
    dogru gorunsun diye: islem suresi = kare-sayma degeri, ilk isleme kadar = eski
    bekleme, islemler arasi = eski postada bos sure kabul edilir.
    Sutun kaymasi yuzunden bozulmus satirlar atilir (None doner).
    """
    try:
        datetime.strptime((r.get("baslangic") or "").strip(), "%H:%M:%S")
    except Exception:
        return None                      # bozuk / kaymis satir
    if not (r.get("vardiya") or "").strip():
        r["vardiya"] = vardiya_adi(_dt_of(r))
    if not (r.get("va_span_sn") or "").strip():
        r["va_span_sn"] = r.get("va_sn") or "0"
        r["nnva_span_sn"] = "0"
        r["nva_bas_sn"] = r.get("bekleme_sn") or "0"
        r["nva_ara_sn"] = r.get("nva_posta_sn") or "0"
        r["nva_son_sn"] = "0"
        r["_eski"] = "1"
    return r


class CycleLogger:
    """Cevrim kayitlarini gunluk CSV dosyalarina yazar/okur."""
    def __init__(self, log_dir=LOG_DIR):
        self.dir = log_dir
        self._migrated = set()
        try:
            os.makedirs(self.dir, exist_ok=True)
        except Exception as e:
            print("[kayit] klasor olusturulamadi:", e)

    def path_for(self, gun):
        return os.path.join(self.dir, f"cevrimler_{gun}.csv")

    def append(self, row):
        p = self.path_for(row["tarih"])
        try:
            self._migrate(p)                       # eski surum dosyasini tasi
            yeni = not os.path.exists(p)
            with open(p, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=CYCLE_FIELDS, delimiter=";",
                                   extrasaction="ignore")
                if yeni:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:
            print("[kayit] yazilamadi:", e)

    def _migrate(self, p):
        """Eski surumle yazilmis CSV'yi yeni sutun duzenine tasir.

        ONEMLI: dosyanin basligi ile yazilacak sutunlar farkliysa satirlar
        YANLIS sutunlara yazilir ve veri bozulur. Bu yuzden ekleme yapmadan
        once basligi kontrol edip dosyayi yeniden yaziyoruz.
        """
        if p in self._migrated or not os.path.exists(p):
            self._migrated.add(p)
            return
        self._migrated.add(p)
        try:
            with open(p, newline="", encoding="utf-8-sig") as f:
                basliklar = next(csv.reader(f, delimiter=";"), [])
            if basliklar == CYCLE_FIELDS:
                return
            with open(p, newline="", encoding="utf-8-sig") as f:
                eski = [_normalize(r) for r in csv.DictReader(f, delimiter=";")]
            eski = [r for r in eski if r is not None]
            yedek = p[:-4] + "_yedek.csv"
            os.replace(p, yedek)
            with open(p, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=CYCLE_FIELDS, delimiter=";",
                                   extrasaction="ignore")
                w.writeheader()
                for r in eski:
                    w.writerow({k: r.get(k, "") for k in CYCLE_FIELDS})
            print(f"[kayit] {os.path.basename(p)} yeni surume tasindi "
                  f"({len(eski)} satir) — eski hali: {os.path.basename(yedek)}")
        except Exception as e:
            print("[kayit] surum tasima hatasi:", e)

    def load(self, gun):
        p = self.path_for(gun)
        if not os.path.exists(p):
            return []
        try:
            with open(p, newline="", encoding="utf-8-sig") as f:
                out = [_normalize(r) for r in csv.DictReader(f, delimiter=";")]
            return [r for r in out if r is not None]
        except Exception as e:
            print("[kayit] okunamadi:", e)
            return []

    def gunler(self):
        try:
            return sorted(fn[10:-4] for fn in os.listdir(self.dir)
                          if fn.startswith("cevrimler_") and fn.endswith(".csv"))
        except Exception:
            return []


def satir_vardiyasi(r):
    return (r.get("vardiya") or "").strip() or vardiya_adi(_dt_of(r))


def gecerli_satirlar(rows, vardiya_zorunlu=True):
    """(uretim satirlari, durus satirlari).

    vardiya_zorunlu=True: vardiya saatleri disindaki kayitlar tamamen atilir.
    False: hepsi alinir (gece testi / mesai disi calisma icin geri adim)."""
    uretim, durus = [], []
    for r in rows:
        if vardiya_zorunlu and not satir_vardiyasi(r):
            continue
        (durus if _f(r.get("cevrim_sn")) > CEVRIM_MAX_SN else uretim).append(r)
    return uretim, durus


def _dt_of(r):
    try:
        return datetime.strptime(f"{r.get('tarih','')} {r.get('baslangic','')}",
                                 "%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime(1970, 1, 1)


def _ozetle(rows):
    """Bir satir kumesinin uretim ozeti."""
    o = dict(adet=0, cevrimler=[], islem=0.0, bekleme=0.0,
             islem_span=0.0, va_span=0.0, nnva_span=0.0,
             nva_bas=0.0, nva_ara=0.0, nva_son=0.0,
             eksik_sayac={}, adim_sure={s: 0.0 for s in IS_CLASSES})
    for r in rows:
        o["adet"] += 1
        o["cevrimler"].append(_f(r.get("cevrim_sn")))
        o["islem"] += _f(r.get("va_sn"))
        o["bekleme"] += _f(r.get("bekleme_sn"))
        # va_span + nnva_span = toplam islem suresi (eski kayitlar dahil)
        o["va_span"] += _f(r.get("va_span_sn"))
        o["nnva_span"] += _f(r.get("nnva_span_sn"))
        o["islem_span"] += _f(r.get("va_span_sn")) + _f(r.get("nnva_span_sn"))
        o["nva_bas"] += _f(r.get("nva_bas_sn"))
        o["nva_ara"] += _f(r.get("nva_ara_sn"))
        o["nva_son"] += _f(r.get("nva_son_sn"))
        for s in (r.get("eksik") or "").split(";"):
            s = s.strip()
            if s:
                o["eksik_sayac"][s] = o["eksik_sayac"].get(s, 0) + 1
        for s, c in STEP_COL.items():
            o["adim_sure"][s] += _f(r.get(f"sure_{c}"))
    n = max(1, o["adet"])
    o["ort_cevrim"]     = sum(o["cevrimler"]) / n
    o["ort_islem"]      = o["islem"] / n
    o["ort_bekleme"]    = o["bekleme"] / n
    o["ort_adim"]       = {s: v / n for s, v in o["adim_sure"].items()}
    o["ort_islem_span"] = o["islem_span"] / n
    o["ort_va_span"]    = o["va_span"] / n          # sadece rapor (VA_RAPOR)
    o["ort_nnva_span"]  = o["nnva_span"] / n        # sadece rapor (VA_RAPOR)
    o["ort_nva_bas"]    = o["nva_bas"] / n
    o["ort_nva_ara"]    = o["nva_ara"] / n
    o["ort_nva_son"]    = o["nva_son"] / n
    top = o["ort_cevrim"]
    o["islem_yuzde"]   = 100.0 * o["ort_islem_span"] / top if top > 0 else 0.0
    o["bos_yuzde"]     = 100.0 - o["islem_yuzde"] if top > 0 else 0.0
    o["va_yuzde"]      = 100.0 * o["ort_va_span"] / top if top > 0 else 0.0
    o["nva_yuzde"]     = 100.0 - o["va_yuzde"] if top > 0 else 0.0
    o["cmin"]          = o["ort_cevrim"] / SEC_PER_CMIN
    o["parca_saat"]    = 3600.0 / o["ort_cevrim"] if o["ort_cevrim"] > 0 else 0.0
    o["parca_dk"]      = 60.0 / o["ort_cevrim"] if o["ort_cevrim"] > 0 else 0.0
    o["medyan"]        = statistics.median(o["cevrimler"]) if o["cevrimler"] else 0.0
    o["en_iyi"]        = min(o["cevrimler"]) if o["cevrimler"] else 0.0
    o["en_kotu"]       = max(o["cevrimler"]) if o["cevrimler"] else 0.0
    return o


_RAPOR_CSS = """
:root{--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--ink3:#77766f;
--line:#e6e5e0;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
--iyi:#0ca30c;--kritik:#d03b3b;}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;
--ink2:#c3c2b7;--ink3:#8f8e85;--line:#2e2e2b;--s1:#3987e5;--s2:#d95926;--s3:#199e70;
--s4:#c98500;}}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
font:14px/1.5 "Segoe UI",system-ui,-apple-system,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:21px;margin:0 0 2px} h2{font-size:15px;margin:30px 0 10px;font-weight:600}
.marka{display:flex;align-items:center;gap:14px;margin-bottom:2px}
.marka-logo{height:56px;width:auto;border-radius:10px}
.marka-alt{font-size:11.5px;color:var(--ink2);letter-spacing:.05em;text-transform:uppercase}
.alt{color:var(--ink2);font-size:13px;margin:0 0 20px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi .card{padding:13px 15px}
.kpi .v{font-size:25px;font-weight:600;letter-spacing:-.02em;line-height:1.15;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi .v .u{font-size:14px;font-weight:500;color:var(--ink2);margin-left:3px}
.kpi .k{font-size:11.5px;color:var(--ink2);text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px}
.kpi .s{font-size:12px;color:var(--ink3);margin-top:3px}
.lgn{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--ink2);margin:0 0 14px}
.lgn i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.cols{display:flex;align-items:flex-end;gap:14px;height:200px;border-bottom:1px solid var(--line);justify-content:space-between}
.cols .c{flex:1 1 0;max-width:64px;display:flex;flex-direction:column;justify-content:flex-end;
gap:2px;min-width:0;height:100%}
.cols .tot{font-size:11px;color:var(--ink2);text-align:center;margin-bottom:3px;white-space:nowrap}
.seg{border-radius:2px;min-height:1px}
.seg.top{border-top-left-radius:4px;border-top-right-radius:4px}
.c:hover .seg{outline:2px solid var(--surface);outline-offset:-1px;filter:brightness(1.08)}
.xax{display:flex;gap:14px;margin-top:6px;justify-content:space-between}
.xax div{flex:1 1 0;max-width:64px;text-align:center;font-size:11px;color:var(--ink3);min-width:0;
overflow:hidden;white-space:nowrap}
.cols.dar{gap:3px}.cols.dar .c{max-width:none}
.xax.dar{gap:3px}.xax.dar div{max-width:none;font-size:10px;overflow:visible}
.box{position:relative;flex:1;width:100%;min-height:0}
.box .wk{position:absolute;left:50%;width:2px;transform:translateX(-50%);background:var(--ink3);
border-radius:1px}
.box .bx{position:absolute;left:0;right:0;background:var(--s1);border-radius:3px;opacity:.9}
.box .md{position:absolute;left:0;right:0;height:2px;background:var(--surface)}
.box .tk{position:absolute;left:25%;right:25%;height:2px;background:var(--ink3);border-radius:1px}
.box:hover .bx{filter:brightness(1.1)}
.hb{display:grid;grid-template-columns:150px 1fr 96px;gap:10px;align-items:center;margin-bottom:7px}
.hb .n{font-size:12.5px;color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hb .t{height:14px;background:linear-gradient(var(--s1),var(--s1));border-radius:2px 4px 4px 2px}
.hb .v{font-size:12.5px;color:var(--ink2);text-align:right}
.tblwrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:12.5px;min-width:min-content}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}
td{white-space:nowrap}
th{color:var(--ink2);font-weight:600;font-size:11px;text-transform:uppercase;
letter-spacing:.03em;white-space:normal;vertical-align:bottom;line-height:1.3}
th:first-child,td:first-child{text-align:left}
tbody tr:hover{background:rgba(127,127,127,.07)}
.ok{color:var(--iyi);font-weight:600}.bad{color:var(--kritik);font-weight:600}
.foot{margin-top:34px;color:var(--ink3);font-size:11.5px;border-top:1px solid var(--line);padding-top:12px}
"""


def _amblem_data():
    """Rapor basligina gomulecek logo (varsa)."""
    try:
        import base64
        for yol in (AMBLEM_PATH, LOGO_PATH):
            if os.path.exists(yol):
                with open(yol, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                tur = "png" if yol.lower().endswith(".png") else "jpeg"
                return f"data:image/{tur};base64,{b64}"
    except Exception:
        pass
    return ""


def _rapor_basligi(gun):
    logo = _amblem_data()
    img = (f'<img src="{logo}" alt="{URUN_ADI}" class="marka-logo">' if logo else "")
    return ('<div class="marka">' + img
            + f'<div><h1>{URUN_ADI} — Posta uretim raporu</h1>'
              f'<div class="marka-alt">{URUN_ALT}</div></div></div>')


def _kpi(k, v, s=""):
    return (f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
            f'<div class="s">{s}</div></div>')


def _stacked(gruplar, seri, birim="sn", ondalik=1, etiket_her=1, dar=False):
    """gruplar: [(etiket, {ad: deger}), ...]   seri: [(ad, css_renk), ...]"""
    if not gruplar:
        return '<p class="alt">Veri yok.</p>'
    enb = max((sum(d.values()) for _, d in gruplar), default=0.0) or 1.0
    cls_c = "cols dar" if dar else "cols"
    cols, xs = [], []
    for gi, (etiket, d) in enumerate(gruplar):
        top = sum(d.values())
        goster = (gi % etiket_her == 0) or (gi == len(gruplar) - 1)
        segs = []
        for i, (ad, renk) in enumerate(seri):
            v = d.get(ad, 0.0)
            if v <= 0:
                continue
            h = 100.0 * v / enb
            cls = "seg top" if i == len(seri) - 1 or all(
                d.get(a, 0.0) <= 0 for a, _ in seri[i + 1:]) else "seg"
            segs.append(f'<div class="{cls}" style="height:{h:.2f}%;background:{renk}" '
                        f'title="{etiket} · {ad}: {nf(v, ondalik)} {birim}"></div>')
        cols.append(f'<div class="c"><div class="tot">'
                    f'{nf(top, ondalik) if goster else ""}</div>'
                    + "".join(reversed(segs)) + "</div>")
        xs.append(f'<div>{etiket if goster else ""}</div>')
    return (f'<div class="{cls_c}">' + "".join(cols) + f'</div><div class="xax'
            + (" dar" if dar else "") + '">' + "".join(xs) + "</div>")


def _hbars(satirlar, birim="sn", renk="var(--s1)", ondalik=1):
    if not satirlar:
        return '<p class="alt">Veri yok.</p>'
    enb = max((v for _, v in satirlar), default=0.0) or 1.0
    out = []
    for ad, v in satirlar:
        w = 100.0 * v / enb
        out.append(f'<div class="hb"><div class="n">{ad}</div>'
                   f'<div><div class="t" style="width:{w:.1f}%;background:{renk}" '
                   f'title="{ad}: {nf(v, ondalik)} {birim}"></div></div>'
                   f'<div class="v">{nf(v, ondalik)} {birim}</div></div>')
    return "".join(out)


# ══ EKSIK ISLEM KLIPLERI ═════════════════════════════════════════════════
# Kameradan gecen kareler surekli bir halka tamponda tutulur. Bir cevrim
# EKSIK kapandiginda son KLIP_SANIYE saniye video olarak diske yazilir;
# operatorun neyi atladigi/yanlis yaptigi izlenerek gorulebilir.
#
# Bellek: kareler kucultulup JPEG olarak saklanir.
#   ornek: 300 sn x 6 fps x ~35 KB  ~=  60 MB
# KLIP KAYDI SIMDILIK KAPALI. Tekrar acmak icin KLIP_AKTIF = True yeter;
# butun altyapi (tampon, eksik penceresi, rapor bolumu) yerinde duruyor.
KLIP_AKTIF   = False
KLIP_KAYDET  = True                                   # eksikte klip al
KLIP_SANIYE  = 90                                     # tampon: son 1,5 dk
KLIP_FPS     = 6                                      # saniyede saklanan kare
KLIP_OLCEK   = 0.6                                    # kare boyutu carpani
KLIP_KALITE  = 75                                     # JPEG kalitesi
KLIP_MAX_MB  = 250                                    # tampon bellek tavani
KLIP_DIR     = os.path.join(LOG_DIR, "klipler")

# Klip TUM cevrimi degil, eksik adimin YAPILMASI GEREKEN araligini kapsar:
# o adimdan onceki ve sonraki islemlerin arasinda kalan bosluk.
KLIP_ONCE    = 3.0        # pencerenin oncesine eklenen pay (sn)
KLIP_SONRA   = 3.0        # sonrasina eklenen pay (sn)
KLIP_MIN_SN  = 6.0        # klip bundan kisa olmasin
KLIP_MAX_SN  = 30.0       # klip bundan uzun olmasin
KLIP_ELLE_SN = 30.0       # "simdi kaydet" butonunun aldigi sure

# Islemlerin gercek sirasi — eksik adimin yerini bulmak icin kullanilir.
# Sahadaki sira farkliysa burayi duzenle.
ISLEM_SIRASI = ["kam_mili koyma", "kep_sokme", "musur_takma", "musur_skma",
                "pim cakma", "silikon_silme"]


class ClipBuffer:
    """Son KLIP_SANIYE saniyelik kareleri bellekte tutan halka tampon."""
    def __init__(self):
        self.q = deque()
        self.bytes = 0
        self.lock = threading.Lock()
        self._last = 0.0

    def feed(self, now, frame):
        # %10 tolerans: kare araliklari tam esitse yuvarlama yuzunden her
        # ikinci kare atlanabiliyor, bu da klip hizini yariya dusuruyordu
        if frame is None or (now - self._last) < 0.9 / max(1, KLIP_FPS):
            return
        self._last = now
        try:
            k = (cv2.resize(frame, None, fx=KLIP_OLCEK, fy=KLIP_OLCEK)
                 if KLIP_OLCEK != 1.0 else frame)
            ok, buf = cv2.imencode(".jpg", k,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), KLIP_KALITE])
            if not ok:
                return
            b = buf.tobytes()
        except Exception:
            return
        with self.lock:
            self.q.append((now, b))
            self.bytes += len(b)
            tavan = KLIP_MAX_MB * 1024 * 1024
            while self.q and ((now - self.q[0][0]) > KLIP_SANIYE or self.bytes > tavan):
                _, eski = self.q.popleft()
                self.bytes -= len(eski)

    def al(self, sn=None):
        """Son sn saniyelik kareler [(zaman, jpeg), ...]"""
        with self.lock:
            if not self.q:
                return []
            son = self.q[-1][0]
            bas = son - (sn if sn else KLIP_SANIYE)
            return [(t, b) for t, b in self.q if t >= bas]

    def aralik(self, bas, bit):
        """Iki mutlak zaman arasindaki kareler."""
        with self.lock:
            return [(t, b) for t, b in self.q if bas <= t <= bit]

    @property
    def sure(self):
        with self.lock:
            return (self.q[-1][0] - self.q[0][0]) if len(self.q) > 1 else 0.0

    @property
    def mb(self):
        return self.bytes / 1024.0 / 1024.0


def eksik_penceresi(cyc, eksik, t_end):
    """Eksik adimin yapilmasi GEREKTIGI zaman araligini tahmin eder.

    Mantik: islem sirasinda o adimdan onceki en yakin GORULEN islemin bitisi
    ile sonraki en yakin gorulen islemin baslangici arasindaki bosluk.
    Ornek: kep sokme 14:22:05'te bitti, silikon silme 14:22:19'da basladi ama
    musur sikma hic gorulmedi -> pencere 14:22:05 - 14:22:19.

    Hicbir komsu yoksa: cevrim icindeki EN UZUN bos aralik kullanilir.
    Doner: (bas, bit, aciklama)
    """
    spans = (cyc or {}).get("spans") or {}
    gorulen = {ad: (lst[0][0], lst[-1][1]) for ad, lst in spans.items() if lst}
    t0 = (cyc or {}).get("t0", t_end)
    sira = [a for a in ISLEM_SIRASI if a in IS_WORK_SET] or list(IS_CLASSES)

    pencereler, aciklamalar = [], []
    for adim in (eksik or []):
        i = sira.index(adim) if adim in sira else -1
        onceki = sonraki = None
        onceki_ad = sonraki_ad = ""
        if i >= 0:
            for j in range(i - 1, -1, -1):
                if sira[j] in gorulen:
                    onceki, onceki_ad = gorulen[sira[j]][1], sira[j]; break
            for j in range(i + 1, len(sira)):
                if sira[j] in gorulen:
                    sonraki, sonraki_ad = gorulen[sira[j]][0], sira[j]; break
        if onceki is not None or sonraki is not None:
            bas = onceki if onceki is not None else t0
            bit = sonraki if sonraki is not None else t_end
            if bit > bas:
                pencereler.append((bas, bit))
                aciklamalar.append(
                    f"{pretty(adim)}: "
                    + (f"{pretty(onceki_ad)} sonrasi" if onceki_ad else "cevrim basi")
                    + " - "
                    + (f"{pretty(sonraki_ad)} oncesi" if sonraki_ad else "cevrim sonu"))
                continue
        # komsu yok -> en uzun bos aralik
        ivs = _merge_spans(spans)
        bosluklar = []
        if ivs:
            if ivs[0][0] > t0:
                bosluklar.append((t0, ivs[0][0]))
            for a1, b1 in zip(ivs, ivs[1:]):
                bosluklar.append((a1[1], b1[0]))
            if t_end > ivs[-1][1]:
                bosluklar.append((ivs[-1][1], t_end))
        if bosluklar:
            bas, bit = max(bosluklar, key=lambda x: x[1] - x[0])
            pencereler.append((bas, bit))
            aciklamalar.append(f"{pretty(adim)}: cevrimdeki en uzun bos aralik")
        else:
            pencereler.append((t0, t_end))
            aciklamalar.append(f"{pretty(adim)}: hicbir islem gorulmedi, tum cevrim")

    if not pencereler:
        return max(t0, t_end - KLIP_MAX_SN), t_end, "cevrim sonu"

    bas = min(p[0] for p in pencereler) - KLIP_ONCE
    bit = max(p[1] for p in pencereler) + KLIP_SONRA
    bas = max(bas, t0 - KLIP_ONCE)
    bit = min(bit, t_end + KLIP_SONRA)
    if bit - bas < KLIP_MIN_SN:                       # cok dar -> genislet
        orta = (bas + bit) / 2.0
        bas, bit = orta - KLIP_MIN_SN / 2, orta + KLIP_MIN_SN / 2
    if bit - bas > KLIP_MAX_SN:
        # cok genis -> SONU koru: eksik adimin yapilmasi icin son firsat,
        # bir sonraki islemin hemen oncesidir
        bas = bit - KLIP_MAX_SN
    return bas, bit, " | ".join(aciklamalar)


def klip_yaz(kareler, yol, baslik="", fps=None):
    """JPEG kareleri mp4 olarak yazar; her kareye saat ve baslik basar."""
    if not kareler:
        return None
    fps = fps or KLIP_FPS
    ilk = cv2.imdecode(np.frombuffer(kareler[0][1], np.uint8), cv2.IMREAD_COLOR)
    if ilk is None:
        return None
    h, w = ilk.shape[:2]
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    vw = cv2.VideoWriter(yol, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        print("[klip] video yazici acilamadi:", yol)
        return None
    t_son = kareler[-1][0]
    for t, b in kareler:
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ov = img.copy()
        cv2.rectangle(ov, (0, h - 26), (w, h), (0, 0, 0), -1)
        img = cv2.addWeighted(ov, 0.55, img, 0.45, 0)
        saat = datetime.fromtimestamp(t).strftime("%H:%M:%S")
        geri = t - t_son                      # cevrim sonuna gore (negatif)
        cv2.putText(img, f"{saat}  ({geri:+.0f} sn)", (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if baslik:
            (tw, _), _ = cv2.getTextSize(baslik, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(img, baslik, (max(8, w - tw - 8), h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    return yol


def klip_adi(t, etiket, ek=""):
    d = datetime.fromtimestamp(t).strftime("%Y-%m-%d_%H-%M-%S")
    ek = "_" + _sanit(ek) if ek else ""
    return os.path.join(KLIP_DIR, f"{etiket}_{d}{ek}.mp4")


def gun_klipleri(gun):
    """Bir gunun klip dosyalari (yeniden eskiye)."""
    if not os.path.isdir(KLIP_DIR):
        return []
    try:
        return sorted((f for f in os.listdir(KLIP_DIR)
                       if f.endswith(".mp4") and gun in f), reverse=True)
    except Exception:
        return []


WEB_HTML = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>__ADI__ — Canli Panel</title>
<style>
/* AI-LINER 6 — canli panel. Tanitim sayfasiyla ayni dil: "void" zemin,
   ucgen takimyildizi, ince/hafif rakamlar, tek camgobegi eylem rengi. */
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#04060a; --bg2:#080c12; --cam:rgba(255,255,255,.028);
  --cam2:rgba(255,255,255,.055); --cizgi:rgba(255,255,255,.07);
  --ink:#e6edf3; --ink2:#a3b1c2; --ink3:#5f7086;
  --cyan:#22d3ee; --mor:#a855f7; --yesil:#22c55e; --sari:#facc15;
  --turuncu:#fb923c; --kirmizi:#f87171;
  --r:20px; --ease:cubic-bezier(.4,0,.2,1);
  --font:"Inter","Segoe UI Variable Display","Segoe UI Variable","Segoe UI",system-ui,sans-serif;
}
html,body{height:100%}
body{background:var(--bg);color:var(--ink);overflow:hidden;
  font:14px/1.5 var(--font);-webkit-font-smoothing:antialiased;font-feature-settings:"tnum" 0}
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(1000px 640px at 8% -14%, rgba(34,211,238,.10), transparent 62%),
    radial-gradient(900px 640px at 96% 4%, rgba(168,85,247,.10), transparent 62%),
    radial-gradient(800px 520px at 58% 118%, rgba(251,146,60,.06), transparent 62%)}
#yildiz{position:fixed;inset:0;width:100%;height:100%;z-index:0;pointer-events:none;opacity:.55}
.uyg{position:relative;z-index:1;height:100vh;display:grid;
  grid-template-rows:auto 1fr;gap:14px;padding:14px 18px 18px}

/* ── ust bar ── */
.bar{display:flex;align-items:center;gap:14px}
.marka{display:flex;align-items:center;gap:12px}
.marka img{height:40px;width:40px;object-fit:cover;border-radius:50%;
  box-shadow:0 0 0 1px rgba(255,255,255,.08),0 0 30px rgba(168,85,247,.28)}
.marka .ad{font-size:19px;font-weight:600;letter-spacing:-.01em;display:flex;align-items:center;gap:8px}
.marka .ad span{background:linear-gradient(92deg,var(--cyan),var(--mor) 60%,var(--turuncu));
  -webkit-background-clip:text;background-clip:text;color:transparent}
.marka .ad i{font-style:normal;font-size:10px;font-weight:600;letter-spacing:.08em;color:var(--bg);
  background:var(--cyan);border-radius:99px;padding:2px 8px}
.marka .alt{font-size:10px;color:var(--ink3);letter-spacing:.14em;text-transform:uppercase;font-weight:600}
.rozetler{display:flex;gap:8px;margin-left:auto;align-items:center;flex-wrap:wrap}
.rozet{display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:999px;
  background:var(--cam);border:1px solid var(--cizgi);backdrop-filter:blur(14px);
  font-size:12px;font-weight:500;color:var(--ink3);white-space:nowrap}
.rozet b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:0;font-size:13px}
.nokta{width:8px;height:8px;border-radius:50%;background:var(--ink3);flex:none}
.nokta.canli{background:var(--yesil);box-shadow:0 0 0 0 rgba(34,197,94,.6);animation:nabiz 2s infinite}
@keyframes nabiz{70%{box-shadow:0 0 0 9px rgba(34,197,94,0)}100%{box-shadow:0 0 0 0 rgba(34,197,94,0)}}
.ikon-btn{width:38px;height:38px;display:grid;place-items:center;border-radius:50%;
  background:var(--cam);border:1px solid var(--cizgi);color:var(--ink2);cursor:pointer;
  font-size:15px;transition:background .25s var(--ease),color .25s,transform .25s var(--ease);text-decoration:none}
.ikon-btn:hover{background:var(--cam2);color:var(--ink);transform:translateY(-1px)}
.ikon-btn.tanitim{width:auto;padding:0 16px;border-radius:99px;font-size:11px;font-weight:600;
  letter-spacing:.08em;text-transform:uppercase;gap:7px;display:flex;align-items:center}
.ikon-btn.tanitim::before{content:"";width:7px;height:7px;transform:rotate(45deg);
  background:linear-gradient(135deg,var(--cyan),var(--mor))}

/* ── yerlesim ── */
.govde{display:grid;grid-template-columns:1fr 380px;gap:14px;min-height:0}
.sol{display:grid;grid-template-rows:1fr auto;gap:14px;min-height:0}
.sag{display:flex;flex-direction:column;gap:12px;min-height:0;overflow-y:auto;
  overflow-x:hidden;padding-right:4px;scrollbar-width:thin;scrollbar-color:var(--cizgi) transparent}
.sag::-webkit-scrollbar{width:6px}
.sag::-webkit-scrollbar-thumb{background:var(--cizgi);border-radius:99px}
.kart{background:linear-gradient(180deg,rgba(255,255,255,.035),rgba(255,255,255,.015));
  border:1px solid var(--cizgi);border-radius:var(--r);
  backdrop-filter:blur(18px);position:relative;overflow:hidden;flex:0 0 auto}
.kart::before{content:"";position:absolute;inset:0 0 auto;height:1px;
  background:linear-gradient(90deg,transparent,rgba(34,211,238,.35),rgba(168,85,247,.35),transparent)}
.kart-bas{display:flex;align-items:center;gap:10px;padding:14px 18px 0;
  font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3);font-weight:600}
.kart-bas .cizgi{flex:1;height:1px;background:var(--cizgi)}

/* ── video ── */
.video{position:relative;border-radius:var(--r);overflow:hidden;background:#000;
  border:1px solid var(--cizgi);display:grid;place-items:center;min-height:0;
  box-shadow:0 30px 80px rgba(0,0,0,.55)}
.video img{width:100%;height:100%;object-fit:contain;display:block}
.hud{position:absolute;top:14px;left:14px;display:flex;gap:8px;flex-wrap:wrap}
.hud .cip{padding:7px 13px;border-radius:99px;font-size:11px;font-weight:600;letter-spacing:.04em;
  background:rgba(4,6,10,.62);border:1px solid rgba(255,255,255,.1);backdrop-filter:blur(12px);
  display:flex;align-items:center;gap:7px}
.video .kose{position:absolute;width:24px;height:24px;border:1.5px solid var(--cyan);
  opacity:.45;pointer-events:none}
.kose.sol-ust{top:12px;left:12px;border-right:0;border-bottom:0;border-radius:8px 0 0 0}
.kose.sag-ust{top:12px;right:12px;border-left:0;border-bottom:0;border-radius:0 8px 0 0;border-color:var(--mor)}
.kose.sol-alt{bottom:12px;left:12px;border-right:0;border-top:0;border-radius:0 0 0 8px;border-color:var(--mor)}
.kose.sag-alt{bottom:12px;right:12px;border-left:0;border-top:0;border-radius:0 0 8px 0;border-color:var(--turuncu)}

/* ── kpi ── */
.kpi{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.kpi .k{background:linear-gradient(180deg,rgba(255,255,255,.035),rgba(255,255,255,.012));
  border:1px solid var(--cizgi);border-radius:18px;
  padding:14px 18px 13px;backdrop-filter:blur(14px);position:relative;overflow:hidden;
  transition:background .25s var(--ease),transform .25s var(--ease)}
.kpi .k:hover{background:var(--cam2);transform:translateY(-2px)}
.kpi .k::after{content:"";position:absolute;left:18px;right:18px;top:0;height:2px;border-radius:0 0 2px 2px;
  background:var(--akt,var(--cyan));opacity:.9}
.kpi .e{font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3);font-weight:600}
.kpi .d{font-size:34px;font-weight:300;letter-spacing:-.045em;line-height:1.2;
  font-variant-numeric:tabular-nums;color:var(--akt,var(--ink))}
.kpi .a{font-size:11.5px;color:var(--ink3);font-variant-numeric:tabular-nums}

/* ── QA ── */
.qa-ust{display:flex;align-items:center;gap:18px;padding:14px 18px 8px}
.halka{position:relative;flex:none;width:104px;height:104px}
.halka svg{transform:rotate(-90deg)}
.halka .yz{position:absolute;inset:0;display:grid;place-items:center;text-align:center}
.halka .sy{font-size:26px;font-weight:300;letter-spacing:-.04em;line-height:1}
.halka .lb{font-size:8.5px;letter-spacing:.16em;color:var(--ink3);font-weight:600;margin-top:4px}
.qa-durum{font-size:32px;font-weight:400;letter-spacing:-.04em;line-height:1.05}
.qa-alt{font-size:12.5px;color:var(--ink2);margin-top:4px;font-weight:300}
.adimlar{padding:4px 14px 16px;display:flex;flex-direction:column;gap:2px}
.adim{display:flex;align-items:center;gap:12px;padding:9px 10px;border-radius:12px;
  transition:background .2s;position:relative;overflow:hidden}
.adim:hover{background:rgba(255,255,255,.03)}
.adim .n{width:9px;height:9px;border-radius:50%;background:var(--cizgi);flex:none;transition:.25s}
.adim .ad{flex:1;font-size:14px;font-weight:300;color:var(--ink2);transition:.2s}
.adim .rz{font-size:11px;color:var(--ink3);letter-spacing:.02em;font-weight:500}
.adim .ilerleme{position:absolute;left:0;bottom:0;height:2px;background:var(--sari);
  transition:width .25s;border-radius:2px}
.adim.tamam .n{background:var(--yesil);box-shadow:0 0 12px rgba(34,197,94,.75)}
.adim.tamam .ad{color:var(--ink);font-weight:400}
.adim.tamam .rz{color:var(--yesil)}
.adim.ilerliyor .n{background:var(--sari);box-shadow:0 0 12px rgba(250,204,21,.7)}
.adim.ilerliyor .ad{color:var(--ink)}
.adim.ilerliyor .rz{color:var(--sari)}
.adim.kilitli{opacity:.45}
.adim.kilitli .n{background:transparent;border:1px solid var(--ink3)}

/* ── motor / bolge ── */
.satir{display:flex;align-items:center;gap:12px;padding:10px 18px}
.satir .ad{flex:1;font-size:13.5px;font-weight:300;color:var(--ink2)}
.satir .dg{font-size:13px;font-weight:600;font-variant-numeric:tabular-nums}
.cubuk{height:4px;border-radius:99px;background:rgba(255,255,255,.06);overflow:hidden;margin:0 18px 14px}
.cubuk i{display:block;height:100%;border-radius:99px;transition:width .3s,background .3s}
.motor-ad{font-size:24px;font-weight:400;letter-spacing:-.03em}
.motor-alt{font-size:11.5px;color:var(--ink3);font-variant-numeric:tabular-nums}

/* ── sparkline ── */
.spark{padding:8px 18px 16px}
.spark canvas{width:100%;height:74px;display:block}

/* ── butonlar ── */
.dugmeler{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:6px 18px 18px}
.btn{padding:11px 12px;border-radius:99px;border:1px solid var(--cizgi);cursor:pointer;
  background:var(--cam);color:var(--ink2);font:inherit;font-size:11.5px;font-weight:600;
  letter-spacing:.05em;text-transform:uppercase;
  transition:background .2s,color .2s,transform .2s var(--ease),box-shadow .25s;
  display:flex;align-items:center;justify-content:center;gap:7px}
.btn:hover{background:var(--cam2);color:var(--ink);transform:translateY(-1px)}
.btn.ana{background:var(--cyan);border-color:var(--cyan);color:#031318}
.btn.ana:hover{background:var(--cyan);color:#031318;box-shadow:0 10px 32px rgba(34,211,238,.3)}
.btn.genis{grid-column:1/-1}
.btn:focus-visible,.ikon-btn:focus-visible{outline:2px solid var(--cyan);outline-offset:2px}

/* ── ayar cekmecesi ── */
.cekmece{position:fixed;top:0;right:0;bottom:0;width:min(410px,92vw);z-index:30;
  background:rgba(6,9,14,.94);backdrop-filter:blur(24px);border-left:1px solid var(--cizgi);
  transform:translateX(102%);transition:transform .35s var(--ease);
  display:flex;flex-direction:column}
.cekmece.acik{transform:none}
.cekmece .ic{overflow:auto;padding:0 20px 24px;scrollbar-width:thin}
.cek-bas{display:flex;align-items:center;gap:10px;padding:20px}
.cek-bas h2{font-size:22px;font-weight:400;letter-spacing:-.03em;flex:1}
.grup{margin-top:18px}
.grup h3{font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--turuncu);
  font-weight:600;margin-bottom:10px}
.ayar{margin-bottom:13px}
.ayar .st{display:flex;justify-content:space-between;font-size:12.5px;font-weight:300;color:var(--ink2);margin-bottom:6px}
.ayar .st b{color:var(--cyan);font-weight:600;font-variant-numeric:tabular-nums}
input[type=range]{width:100%;height:4px;border-radius:99px;appearance:none;
  background:linear-gradient(90deg,var(--cyan),var(--mor) var(--p,50%),rgba(255,255,255,.08) var(--p,50%));
  outline:0;cursor:pointer}
input[type=range]::-webkit-slider-thumb{appearance:none;width:16px;height:16px;border-radius:50%;
  background:#fff;box-shadow:0 0 0 4px rgba(34,211,238,.18),0 0 14px rgba(34,211,238,.7);cursor:pointer}
.anahtar{display:flex;align-items:center;gap:12px;padding:9px 0;cursor:pointer;font-size:13.5px;font-weight:300;color:var(--ink2)}
.anahtar .kutu{width:38px;height:21px;border-radius:99px;background:rgba(255,255,255,.09);
  position:relative;transition:.22s;flex:none}
.anahtar .kutu::after{content:"";position:absolute;top:2.5px;left:2.5px;width:16px;height:16px;
  border-radius:50%;background:#7d8da1;transition:.22s var(--ease)}
.anahtar.acik .kutu{background:linear-gradient(90deg,var(--cyan),var(--mor))}
.anahtar.acik .kutu::after{left:19.5px;background:#fff}
.anahtar.acik{color:var(--ink)}

/* ── bildirim ── */
.toastlar{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);z-index:40;
  display:flex;flex-direction:column;gap:9px;align-items:center}
.toast{padding:14px 24px;border-radius:99px;background:rgba(6,9,14,.92);
  border:1px solid var(--cizgi);backdrop-filter:blur(18px);font-size:13.5px;font-weight:600;
  display:flex;align-items:center;gap:12px;animation:gir .4s var(--ease);
  box-shadow:0 18px 50px rgba(0,0,0,.55)}
@keyframes gir{from{opacity:0;transform:translateY(22px);filter:blur(4px)}}
.toast.gecti{border-color:rgba(34,197,94,.5);color:var(--yesil)}
.toast.eksik{border-color:rgba(251,146,60,.5);color:var(--turuncu)}
.toast small{color:var(--ink3);font-weight:400}
.kapali .uyg{filter:blur(3px);opacity:.5;pointer-events:none}
@media(max-width:1100px){
  body{overflow:auto}
  .uyg{height:auto;min-height:100vh}
  .govde{grid-template-columns:1fr}
  .sol{grid-template-rows:auto auto}
  .video{aspect-ratio:16/9;min-height:200px}
  .sag{max-height:none;overflow:visible;padding-right:0}
  .kpi{grid-template-columns:repeat(2,1fr)}
  .kpi .d{font-size:28px}
  .marka .alt{display:none}
  .bar{flex-wrap:wrap}
  .hud{top:10px;left:10px;gap:6px}.hud .cip{font-size:10px;padding:5px 9px}
}
@media(max-width:520px){.ikon-btn.tanitim{display:none}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<canvas id="yildiz" aria-hidden="true"></canvas>
<div class="uyg">
  <div class="bar">
    <div class="marka">__LOGO__
      <div><div class="ad"><span>__ADI__</span><i>v__SURUM__</i></div><div class="alt">__ALT__</div></div></div>
    <div class="rozetler">
      <div class="rozet"><span class="nokta" id="nokta"></span><b id="fps">—</b> fps</div>
      <div class="rozet">Kisi <b id="kisi">0</b></div>
      <div class="rozet" id="kaynak-r" title="">Kaynak <b id="kaynak">—</b></div>
      <a class="ikon-btn tanitim" href="/tanitim/" target="_blank" rel="noopener" title="Uygulamayı anlatan tanıtım sayfası">Tanıtım</a>
      <div class="ikon-btn" id="btn-duraklat" title="Duraklat / Devam">⏸</div>
      <div class="ikon-btn" id="btn-tam" title="Tam ekran">⛶</div>
      <div class="ikon-btn" id="btn-ayar" title="Ayarlar">⚙</div>
    </div>
  </div>

  <div class="govde">
    <div class="sol">
      <div class="video">
        <img id="cam" src="/video" alt="canli goruntu">
        <i class="kose sol-ust"></i><i class="kose sag-ust"></i>
        <i class="kose sol-alt"></i><i class="kose sag-alt"></i>
        <div class="hud">
          <div class="cip" id="hud-qa"><span class="nokta"></span><span>—</span></div>
          <div class="cip" id="hud-motor">MOTOR —</div>
          <div class="cip" id="hud-aktif">— sn</div>
        </div>
      </div>
      <div class="kpi" id="kpi"></div>
    </div>

    <div class="sag">
      <div class="kart">
        <div class="kart-bas">ISLEM KONTROL LISTESI<span class="cizgi"></span></div>
        <div class="qa-ust">
          <div class="halka">
            <svg width="104" height="104" viewBox="0 0 104 104">
              <defs><linearGradient id="gr" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="#22d3ee"/><stop offset=".6" stop-color="#a855f7"/><stop offset="1" stop-color="#fb923c"/>
              </linearGradient></defs>
              <circle cx="52" cy="52" r="44" fill="none" stroke="rgba(255,255,255,.06)"
                      stroke-width="7"/>
              <circle id="yay" cx="52" cy="52" r="44" fill="none" stroke="url(#gr)"
                      stroke-width="7" stroke-linecap="round" stroke-dasharray="276.5"
                      stroke-dashoffset="276.5"
                      style="transition:stroke-dashoffset .45s cubic-bezier(.3,.9,.3,1)"/>
            </svg>
            <div class="yz"><div><div class="sy" id="halka-sy">0/5</div>
              <div class="lb">ADIM</div></div></div>
          </div>
          <div><div class="qa-durum" id="qa-durum">—</div>
            <div class="qa-alt" id="qa-alt">baglaniyor...</div></div>
        </div>
        <div class="adimlar" id="adimlar"></div>
      </div>

      <div class="kart">
        <div class="kart-bas">MOTOR CEVRIMI<span class="cizgi"></span></div>
        <div class="satir"><span class="nokta" id="motor-nokta"></span>
          <div style="flex:1"><div class="motor-ad" id="motor-ad">—</div>
            <div class="motor-alt" id="motor-alt">—</div></div></div>
        <div class="cubuk"><i id="motor-bar" style="width:0"></i></div>
      </div>

      <div class="kart" id="bolge-kart">
        <div class="kart-bas">BOLGELER<span class="cizgi"></span></div>
        <div id="bolgeler"></div>
      </div>

      <div class="kart">
        <div class="kart-bas">SON CEVRIMLER<span class="cizgi"></span></div>
        <div class="spark"><canvas id="spark"></canvas></div>
      </div>

      <div class="kart">
        <div class="kart-bas">ISLEMLER<span class="cizgi"></span></div>
        <div class="dugmeler">
          <button class="btn" data-k="qa_sifirla">⭯ QA sifirla</button>
          <button class="btn" data-k="bolge_sifirla">▣ Bolge sifirla</button>
          <button class="btn" data-k="snapshot">⤓ Anlik goruntu</button>
          <button class="btn" data-k="rapor">◱ Rapor</button>
          <button class="btn ana genis" data-k="rapor_ai">✦ Rapor + AI analisti</button>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="cekmece" id="cekmece">
  <div class="cek-bas"><h2>Ayarlar</h2>
    <div class="ikon-btn" id="btn-kapat">✕</div></div>
  <div class="ic" id="ayarlar"></div>
</div>
<div class="toastlar" id="toastlar"></div>

<script>
const $ = s => document.querySelector(s);
// ── sure birimi (Python tarafindan gonderilir) ──
let BR = "cmin", BK = 1/0.6;          // etiket ve saniye->birim carpani
const S = (sn, ond=0) => (sn * BK).toFixed(ond).replace(".", ",");   // sayi
const SB = (sn, ond=0) => S(sn, ond) + " " + BR;                     // sayi + birim
const KPI = [["son","SON CEVRIM","--cyan"],["ort","GUN ORT.",""],["hiz","HIZ",""],
             ["adet","PARCA",""]];
const AYARLAR = [
 ["Tespit esikleri", [
   ["conf_qa","Genel islem conf",0.01,0.99,0.01,2],
   ["conf_kam","Kam mili conf",0.01,0.99,0.01,2],
   ["dwell_kam","Kam mili onay (sn)",0,3,0.1,1],
   ["conf_musur","Musur sikma conf",0.01,0.99,0.01,2],
   ["dwell_musur","Musur sikma onay (sn)",0,3,0.1,1]]],
 ["Motor cevrimi", [
   ["conf_motor","Motor conf",0.01,0.99,0.01,2],
   ["motor_move","Cikis kayma esigi",0.02,0.5,0.01,2],
   ["motor_hold","Okluzyon toleransi (sn)",0.5,20,0.5,1]]],
 ["Goruntu", [
   ["yuz_kabalik","Mozaik kabaligi",6,30,1,0],
   ["frame_skip","Kare atlama",0,5,1,0],
   ["alpha","Maske saydamligi",0,1,0.05,2]]],
];
const ANAHTARLAR = [["yuz_bulanik","Kisileri gizle (KVKK)"],["show_pose","Iskelet"],
  ["show_zones","Bolgeleri goster"],["show_motor_gate","Motor gostergesi"],
  ["rapor_va","Raporda VA/NVA ayrimi"]];

// ── kurulum ──
$("#kpi").innerHTML = KPI.map(([k,e]) =>
  `<div class="k" id="kpi-${k}"><div class="e">${e}</div>
   <div class="d" id="kpi-${k}-d">—</div><div class="a" id="kpi-${k}-a"></div></div>`).join("");

let ayarKur = false;
function ayarlariKur(d){
  if (ayarKur || !d.ayarlar) return; ayarKur = true;
  let h = "";
  for (const [grup, liste] of AYARLAR){
    h += `<div class="grup"><h3>${grup}</h3>`;
    for (const [ad, etiket, mn, mx, adim, bas] of liste){
      const v = d.ayarlar[ad] ?? mn;
      h += `<div class="ayar"><div class="st"><span>${etiket}</span>
        <b id="v-${ad}">${(+v).toFixed(bas)}</b></div>
        <input type="range" id="s-${ad}" min="${mn}" max="${mx}" step="${adim}"
               value="${v}" data-bas="${bas}"></div>`;
    }
    h += "</div>";
  }
  h += `<div class="grup"><h3>Gorunum</h3>` + ANAHTARLAR.map(([ad,et]) =>
    `<div class="anahtar ${d.ayarlar[ad] ? "acik" : ""}" id="a-${ad}" data-ad="${ad}">
       <div class="kutu"></div><span>${et}</span></div>`).join("") + "</div>";
  $("#ayarlar").innerHTML = h;
  document.querySelectorAll('#ayarlar input[type=range]').forEach(el => {
    const oran = () => {
      const p = (el.value - el.min) / (el.max - el.min) * 100;
      el.style.setProperty("--p", p + "%");
    };
    oran();
    el.addEventListener("input", () => {
      oran();
      const ad = el.id.slice(2), bas = +el.dataset.bas;
      $("#v-" + ad).textContent = (+el.value).toFixed(bas);
      gonder({komut:"ayar", ad, deger:+el.value});
    });
  });
  document.querySelectorAll("#ayarlar .anahtar").forEach(el =>
    el.addEventListener("click", () => {
      el.classList.toggle("acik");
      gonder({komut:"ayar", ad:el.dataset.ad, deger:el.classList.contains("acik")?1:0});
    }));
}

function gonder(veri){
  fetch("/api/komut", {method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify(veri)}).catch(()=>{});
}
document.querySelectorAll(".dugmeler .btn").forEach(b =>
  b.addEventListener("click", () => { gonder({komut:b.dataset.k}); toast(b.textContent.trim(), ""); }));
$("#btn-duraklat").onclick = () => gonder({komut:"duraklat"});
$("#btn-ayar").onclick = () => $("#cekmece").classList.add("acik");
$("#btn-kapat").onclick = () => $("#cekmece").classList.remove("acik");
$("#btn-tam").onclick = () => document.fullscreenElement
  ? document.exitFullscreen() : document.documentElement.requestFullscreen();

function toast(metin, tur, alt){
  const t = document.createElement("div");
  t.className = "toast " + (tur||"");
  t.innerHTML = `<span>${metin}</span>` + (alt ? `<small>${alt}</small>` : "");
  $("#toastlar").appendChild(t);
  setTimeout(() => { t.style.transition=".4s"; t.style.opacity=0;
    t.style.transform="translateY(14px)"; setTimeout(()=>t.remove(),400); }, 3600);
}

// ── cizim ──
const RENK = {tamam:"var(--yesil)", ilerliyor:"var(--sari)", kilitli:"var(--ink3)"};
let sonCevrimId = null;

function render(d){
  if (d.birim){ BR = d.birim; BK = d.birim_kat || 1; }
  $("#fps").textContent = (d.fps||0).toFixed(1);
  $("#nokta").className = "nokta" + ((d.fps||0) > 0.5 ? " canli" : "");
  $("#kisi").textContent = d.kisi ?? 0;
  $("#kaynak").textContent = d.kaynak_kisa || "—";
  $("#kaynak-r").title = d.kaynak || "";
  $("#btn-duraklat").textContent = d.duraklat ? "▶" : "⏸";

  // QA
  const q = d.qa || {}, oran = (q.ok||0) / Math.max(1, q.toplam||1);
  $("#yay").style.strokeDashoffset = 276.5 * (1 - oran);
  $("#halka-sy").textContent = `${q.ok||0}/${q.toplam||0}`;
  const dr = {GECTI:"var(--yesil)", EKSIK:"var(--turuncu)", TAMAM:"var(--yesil)",
              BEKLIYOR:"var(--sari)"}[q.durum] || "var(--ink2)";
  $("#qa-durum").textContent = q.durum || "—";
  $("#qa-durum").style.color = dr;
  $("#qa-alt").textContent = q.alt || "";
  $("#adimlar").innerHTML = (q.adimlar||[]).map(a =>
    `<div class="adim ${a.durum}"><span class="n"></span>
      <span class="ad">${a.ad}</span><span class="rz">${a.rozet||""}</span>
      ${a.oran ? `<i class="ilerleme" style="width:${a.oran*100}%"></i>` : ""}</div>`).join("");
  const hq = $("#hud-qa");
  hq.querySelector("span:last-child").textContent = `${q.durum||""} ${q.ok||0}/${q.toplam||0}`;
  hq.querySelector(".nokta").style.background = dr;
  hq.style.borderColor = dr;

  // motor
  const m = d.motor || {};
  const mr = {"POSTADA":"var(--yesil)","YERLESIYOR":"var(--sari)",
              "GORUNMUYOR":"var(--turuncu)","CIKTI":"var(--mor)"}[m.durum]||"var(--ink3)";
  $("#motor-ad").textContent = m.durum || "—";
  $("#motor-ad").style.color = mr;
  $("#motor-nokta").style.background = mr;
  $("#motor-nokta").className = "nokta" + (m.durum === "POSTADA" ? " canli" : "");
  $("#motor-alt").textContent =
    `kayma ${(m.kayma||0).toFixed(3)} / ${(m.esik||0).toFixed(2)}` +
    (m.gorunmuyor > 0.2 ? ` · gorunmuyor ${m.gorunmuyor.toFixed(1)} sn` : "");
  const mo = Math.min(1, (m.kayma||0) / Math.max(1e-6, m.esik||0.1));
  $("#motor-bar").style.width = (mo*100) + "%";
  $("#motor-bar").style.background = mo > .7 ? "var(--turuncu)" : "var(--cyan)";
  $("#hud-motor").textContent = "MOTOR " + (m.durum||"—");

  // aktif cevrim
  const ak = d.aktif || {};
  $("#hud-aktif").textContent = ak.sn > 0
    ? `${SB(ak.sn)} · islem ${SB(ak.islem||0, 1)}` : "motor bekleniyor";

  // bolgeler
  const bol = d.bolgeler || [];
  $("#bolge-kart").style.display = bol.length ? "" : "none";
  $("#bolgeler").innerHTML = bol.map(b => {
    const c = b.durum === "tamam" ? "var(--yesil)"
            : b.durum === "ilerliyor" ? "var(--sari)" : "var(--ink3)";
    return `<div class="satir"><span class="nokta" style="background:${c}"></span>
      <span class="ad">${b.ad}</span>
      <span class="dg" style="color:${c};font-size:11px">${b.etiket||""}</span></div>`;
  }).join("");

  // KPI
  const g = d.gun || {};
  const kpiV = {
    son:[g.son ? SB(g.son) : "—", g.son ? g.son.toFixed(1)+" sn" : "", "var(--cyan)"],
    ort:[g.ort ? SB(g.ort) : "—", g.ort ? g.ort.toFixed(1)+" sn" : "", ""],
    hiz:[g.hiz ? g.hiz.toFixed(1) : "—", "parca/saat", ""],
    adet:[g.adet ?? "—", "bu saat " + (g.bu_saat ?? 0), ""],
  };
  for (const [k,[d1,a1,c]] of Object.entries(kpiV)){
    $("#kpi-"+k+"-d").textContent = d1;
    $("#kpi-"+k+"-a").textContent = a1;
    $("#kpi-"+k).style.setProperty("--akt", c || "var(--ink)");
  }

  sparkCiz(d.son_cevrimler || [], g.ort || 0);
  ayarlariKur(d);

  if (d.son_cevrim_id && d.son_cevrim_id !== sonCevrimId){
    if (sonCevrimId !== null){
      const s = d.son_cevrim || {};
      toast(s.sonuc === "GECTI" ? "✓ GECTI" : "⚠ EKSIK",
            s.sonuc === "GECTI" ? "gecti" : "eksik",
            SB(s.sn||0) + (s.eksik ? " · " + s.eksik : ""));
    }
    sonCevrimId = d.son_cevrim_id;
  }
}

function sparkCiz(veri, hedef){
  const c = $("#spark"), o = c.getContext("2d");
  const w = c.clientWidth, h = 74;
  if (c.width !== w * devicePixelRatio){
    c.width = w * devicePixelRatio; c.height = h * devicePixelRatio;
    o.scale(devicePixelRatio, devicePixelRatio);
  }
  o.clearRect(0, 0, w, h);
  if (!veri.length){
    o.fillStyle = "rgba(147,167,189,.5)"; o.font = "12px sans-serif";
    o.fillText("henuz cevrim yok", 4, 40); return;
  }
  const n = veri.length, mx = Math.max(...veri.map(v => v.sn), hedef) * 1.12 || 1;
  c.title = "son " + n + " cevrim · en uzun " + SB(mx / 1.12);
  const bw = Math.max(2, w / n - 3);
  veri.forEach((v, i) => {
    const x = i * (w / n), y = h - (v.sn / mx) * (h - 12);
    o.fillStyle = v.sonuc === "GECTI" ? "rgba(34,197,94,.85)" : "rgba(251,146,60,.9)";
    o.beginPath();
    o.roundRect(x, y, bw, h - y, [3, 3, 0, 0]);
    o.fill();
  });
  if (hedef > 0){
    const y = h - (hedef / mx) * (h - 12);
    o.strokeStyle = "rgba(34,211,238,.55)"; o.setLineDash([4, 4]); o.lineWidth = 1;
    o.beginPath(); o.moveTo(0, y); o.lineTo(w, y); o.stroke(); o.setLineDash([]);
    o.fillStyle = "rgba(34,211,238,.85)"; o.font = "10px sans-serif";
    o.fillText("ort " + SB(hedef), 2, Math.max(9, y - 3));
  }
}


// ── arka plan: ucgen takimyildizi (sadece boyut degisince bir kez cizilir) ──
(function(){
  const c = document.getElementById("yildiz"); if (!c) return;
  const R = ["#22d3ee","#3b82f6","#8b5cf6","#a855f7","#ec4899","#fb923c","#facc15"];
  function ciz(){
    const d = Math.min(devicePixelRatio || 1, 2), w = innerWidth, h = innerHeight;
    c.width = w * d; c.height = h * d; const o = c.getContext("2d"); o.setTransform(d,0,0,d,0,0);
    let s = 7; const r = () => (s = (s * 16807) % 2147483647) / 2147483647;
    const n = Math.round(w * h / 5200); o.lineWidth = 1;
    for (let i = 0; i < n; i++){
      const x = r() * w, y = r() * h, z = 1.2 + r() * 2.4, a = r() * 6.28;
      o.globalAlpha = .18 + r() * .35; o.strokeStyle = R[(x / w * R.length) | 0];
      o.beginPath();
      for (let k = 0; k < 3; k++){ const t = a + k * 2.094; o[k ? "lineTo" : "moveTo"](x + Math.cos(t) * z, y + Math.sin(t) * z); }
      o.closePath(); o.stroke();
    }
  }
  ciz(); let z; addEventListener("resize", () => { clearTimeout(z); z = setTimeout(ciz, 200); });
})();

// ── klavye kisayollari ──
addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const k = e.key.toLowerCase();
  if (k === " "){ e.preventDefault(); gonder({komut:"duraklat"}); }
  else if (k === "r"){ gonder({komut:"qa_sifirla"}); toast("QA sifirlandi",""); }
  else if (k === "b"){ gonder({komut:"bolge_sifirla"}); toast("Bolgeler sifirlandi",""); }
  else if (k === "s"){ gonder({komut:"snapshot"}); toast("Anlik goruntu alindi",""); }
  else if (k === "f"){ $("#btn-tam").click(); }
  else if (k === "a"){ $("#cekmece").classList.toggle("acik"); }
  else if (k === "escape"){ $("#cekmece").classList.remove("acik"); }
});

// ── baglanti gozcusu ──
let sonVeri = Date.now(), kopukBildirildi = false;
setInterval(() => {
  const gecen = Date.now() - sonVeri;
  if (gecen > 3000){
    $("#nokta").className = "nokta";
    $("#nokta").style.background = "var(--kirmizi)";
    if (!kopukBildirildi){ kopukBildirildi = true;
      toast("Baglanti koptu", "eksik", "yeniden deneniyor..."); }
  } else if (kopukBildirildi){
    kopukBildirildi = false; $("#nokta").style.background = "";
    toast("Baglanti kuruldu", "gecti");
  }
}, 1000);

// ── baglanti: SSE, olmazsa yoklama ──
let es;
function bagla(){
  try{
    es = new EventSource("/durum");
    es.onmessage = e => { sonVeri = Date.now();
      try { render(JSON.parse(e.data)); } catch(x){} };
    es.onerror = () => { es.close(); setTimeout(bagla, 2000); };
  }catch(x){
    setInterval(() => fetch("/durum.json").then(r=>r.json()).then(render).catch(()=>{}), 400);
  }
}
bagla();
addEventListener("resize", () => sparkCiz([], 0));
</script></body></html>
"""


# ══ WEB PANEL ════════════════════════════════════════════════════════════
# Program icinde kucuk bir HTTP sunucusu calisir; panel tarayicida acilir.
#   /            -> arayuz (tek dosya HTML/CSS/JS, internet gerektirmez)
#   /video       -> MJPEG canli goruntu
#   /durum       -> SSE ile anlik durum akisi (JSON)
#   /api/komut   -> POST: duraklat, qa_sifirla, ayar, rapor...
# Ayni agdaki tablet/telefondan da acilir: http://<bilgisayar-ip>:8800
WEB_PANEL  = True
WEB_PORT   = 8800
WEB_FPS    = 12          # MJPEG kare hizi
WEB_GENIS  = 1100        # yayindaki kare genisligi (piksel)
WEB_KALITE = 72          # JPEG kalitesi
WEB_DURUM_HZ = 4         # durum sozlugu saniyede kac kez hazirlansin

# Tk penceresindeki canli goruntu, kareyi PhotoImage'a cevirdigi icin
# pahalidir. Web paneli varken bunu kismak (veya kapatmak) FPS'i belirgin
# olcude arttirir. 0 = Tk'de video gosterme (sunucu modu).
TK_VIDEO_FPS = 8


def _yerel_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class WebPanel:
    """Canli paneli tarayiciya sunar (Tk penceresine ek olarak)."""

    def __init__(self, app, port=WEB_PORT):
        self.app = app
        self.port = port
        self.kare = None            # son BGR kare (ham)
        self.kare_no = 0
        self.jpg = None             # kodlanmis son kare (TUM izleyiciler paylasir)
        self.jpg_no = 0
        self.izleyici = 0           # acik MJPEG baglantisi sayisi
        self.dinleyici = 0          # acik durum (SSE) baglantisi sayisi
        self.son_istek = 0.0        # son /durum.json istegi (yoklama modu)
        self.kilit = threading.Lock()
        self.kosul = threading.Condition(self.kilit)
        self.jkosul = threading.Condition(threading.Lock())
        self.durum = {}
        self.srv = None
        self.html = self._html()
        threading.Thread(target=self._kodlayici, daemon=True).start()

    def _kodlayici(self):
        """Kareyi SADECE BIR KEZ kodlar; kac izleyici olursa olsun tek is.
        Izleyici yokken hic calismaz -> bos yere CPU harcanmaz."""
        son = -1
        while True:
            try:
                if self.izleyici <= 0:
                    time.sleep(0.25); continue
                with self.kosul:
                    if self.kare_no == son or self.kare is None:
                        self.kosul.wait(0.5)
                    if self.kare is None or self.kare_no == son:
                        continue
                    son = self.kare_no
                    kare = self.kare
                h, w = kare.shape[:2]
                if w > WEB_GENIS:
                    o = WEB_GENIS / w
                    kare = cv2.resize(kare, (WEB_GENIS, int(h * o)),
                                      interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", kare,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), WEB_KALITE])
                if not ok:
                    continue
                with self.jkosul:
                    self.jpg = buf.tobytes()
                    self.jpg_no += 1
                    self.jkosul.notify_all()
                time.sleep(1.0 / max(1, WEB_FPS))
            except Exception:
                time.sleep(0.3)

    # ── sayfa ──
    def _html(self):
        try:
            import base64
            logo = ""
            for yol in (os.path.join(TANITIM_DIR, "assets", "amblem.webp"), AMBLEM_PATH, LOGO_PATH):
                if os.path.exists(yol):
                    with open(yol, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    tur = "webp" if yol.endswith(".webp") else "png"
                    logo = f'<img src="data:image/{tur};base64,{b64}" alt="">'
                    break
        except Exception:
            logo = ""
        return (WEB_HTML.replace("__ADI__", URUN_ADI)
                        .replace("__ALT__", URUN_ALT)
                        .replace("__SURUM__", SURUM)
                        .replace("__LOGO__", logo))

    # ── worker'dan cagrilir ──
    def kare_ver(self, kare):
        if kare is None or self.izleyici <= 0:
            return                       # kimse izlemiyorsa hic ugrasma
        with self.kosul:
            self.kare = kare
            self.kare_no += 1
            self.kosul.notify_all()

    def durum_ver(self, d):
        self.durum = d

    @property
    def izlenen(self):
        """Paneli acan var mi? (SSE baglantisi ya da yakin zamanda yoklama)"""
        return self.dinleyici > 0 or (time.time() - self.son_istek) < 5.0

    # ── sunucu ──
    def basla(self):
        from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
        panel = self

        class Isleyici(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *a): pass

            def _bas(self, kod=200, tur="text/html; charset=utf-8", uzunluk=None, ek=None):
                self.send_response(kod)
                self.send_header("Content-Type", tur)
                if uzunluk is not None:
                    self.send_header("Content-Length", str(uzunluk))
                for k, v in (ek or {}).items():
                    self.send_header(k, v)
                self.end_headers()

            def do_GET(self):
                yol = self.path.split("?")[0]
                if yol in ("/", "/index.html"):
                    g = panel.html.encode("utf-8")
                    self._bas(200, "text/html; charset=utf-8", len(g)); self.wfile.write(g)
                elif yol == "/video":
                    self._video()
                elif yol == "/durum":
                    self._sse()
                elif yol == "/durum.json":
                    panel.son_istek = time.time()
                    g = json.dumps(panel.durum).encode("utf-8")
                    self._bas(200, "application/json", len(g),
                              {"Cache-Control": "no-store"}); self.wfile.write(g)
                elif yol == "/tanitim":
                    self._bas(301, "text/plain", 0, {"Location": "/tanitim/"})
                elif yol.startswith("/tanitim/"):
                    self._tanitim(yol[len("/tanitim/"):] or "index.html")
                else:
                    self._dosya(yol)

            def _tanitim(self, ad):
                """tanitim/ klasorundeki sayfa ve varliklar (video icin Range destekli)."""
                from urllib.parse import unquote
                kok = os.path.realpath(TANITIM_DIR)
                tam = os.path.realpath(os.path.join(kok, unquote(ad)))
                if not tam.startswith(kok + os.sep) or not os.path.isfile(tam):
                    self._bas(404, "text/plain", 9); self.wfile.write(b"bulunamad"); return
                turler = {".html": "text/html; charset=utf-8", ".webp": "image/webp",
                          ".png": "image/png", ".mp4": "video/mp4", ".js": "text/javascript",
                          ".css": "text/css", ".svg": "image/svg+xml", ".json": "application/json"}
                tur = turler.get(os.path.splitext(tam)[1].lower(), "application/octet-stream")
                boyut = os.path.getsize(tam)
                bas, son = 0, boyut - 1
                aralik = self.headers.get("Range", "")
                try:
                    if aralik.startswith("bytes="):
                        a, _, b = aralik[6:].split(",")[0].partition("-")
                        if a: bas = int(a); son = int(b) if b else son
                        else: bas = max(0, boyut - int(b))
                        son = min(son, boyut - 1)
                except ValueError:
                    bas, son, aralik = 0, boyut - 1, ""
                ek = {"Accept-Ranges": "bytes",
                      "Cache-Control": "no-cache" if tur.startswith("text/html") else "max-age=3600"}
                try:
                    with open(tam, "rb") as f:
                        f.seek(bas)
                        if aralik:
                            ek["Content-Range"] = f"bytes {bas}-{son}/{boyut}"
                            self._bas(206, tur, son - bas + 1, ek)
                        else:
                            self._bas(200, tur, boyut, ek)
                        kalan = son - bas + 1
                        while kalan > 0:
                            g = f.read(min(65536, kalan))
                            if not g: break
                            self.wfile.write(g); kalan -= len(g)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def do_POST(self):
                if self.path.split("?")[0] != "/api/komut":
                    self._bas(404, "text/plain", 0); return
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    veri = json.loads(self.rfile.read(n) or b"{}")
                    panel.app.web_komut(veri)
                    g = b'{"ok":true}'
                except Exception as e:
                    g = json.dumps({"ok": False, "hata": str(e)}).encode()
                self._bas(200, "application/json", len(g)); self.wfile.write(g)

            def _dosya(self, yol):
                """kayitlar/ altindaki rapor ve dosyalar."""
                ad = os.path.normpath(yol.lstrip("/")).replace("\\", "/")
                tam = os.path.join(panel.app.logger.dir, os.path.basename(ad))
                if not os.path.exists(tam):
                    self._bas(404, "text/plain", 9); self.wfile.write(b"bulunamad")
                    return
                turler = {".html": "text/html; charset=utf-8", ".csv": "text/csv",
                          ".mp4": "video/mp4", ".png": "image/png", ".txt": "text/plain"}
                tur = turler.get(os.path.splitext(tam)[1].lower(), "application/octet-stream")
                with open(tam, "rb") as f:
                    g = f.read()
                self._bas(200, tur, len(g)); self.wfile.write(g)

            def _video(self):
                sinir = "aiframe"
                self.send_response(200)
                self.send_header("Content-Type",
                                 f"multipart/x-mixed-replace; boundary={sinir}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                panel.izleyici += 1
                son = -1
                try:
                    while panel.app.alive:
                        with panel.jkosul:
                            if panel.jpg_no == son:
                                panel.jkosul.wait(1.0)
                            if panel.jpg is None or panel.jpg_no == son:
                                continue
                            son = panel.jpg_no
                            g = panel.jpg          # hazir kodlanmis kare
                        self.wfile.write(f"--{sinir}\r\nContent-Type: image/jpeg\r\n"
                                         f"Content-Length: {len(g)}\r\n\r\n".encode())
                        self.wfile.write(g); self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    panel.izleyici = max(0, panel.izleyici - 1)

            def _sse(self):
                panel.dinleyici += 1
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    while panel.app.alive:
                        g = json.dumps(panel.durum, ensure_ascii=False)
                        self.wfile.write(f"data: {g}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        time.sleep(0.25)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    panel.dinleyici = max(0, panel.dinleyici - 1)

        for p in range(self.port, self.port + 20):
            try:
                self.srv = ThreadingHTTPServer(("0.0.0.0", p), Isleyici)
            except OSError:
                continue
            self.srv.daemon_threads = True
            self.port = p
            threading.Thread(target=self.srv.serve_forever, daemon=True).start()
            self.adres = f"http://{_yerel_ip()}:{p}"
            print(f"\n  ┌─ WEB PANEL ─────────────────────────────────\n"
                  f"  │  {self.adres}\n"
                  f"  │  ayni agdaki tablet/telefondan da acilir\n"
                  f"  └─────────────────────────────────────────────\n")
            return p
        print("[web] uygun port bulunamadi")
        return None


# ══ RAPORDAKI YAPAY ZEKA ANALISTI (Gemini) ═══════════════════════════════
# Rapor sayfasina bir sohbet paneli gomulur. Panel, raporun TUM hesaplanmis
# verisini JSON olarak tasir ve sorulari Gemini'ye gonderir; model grafigi
# degil gercek sayilari yorumlar.
#
# API anahtari icin iki yol:
#   1) BASE_DIR/gemini_key.txt dosyasina anahtari yaz -> rapora gomulur
#      (dosyayi baskasiyla paylasma, anahtar icinde olur)
#   2) Dosya yoksa sayfa bir kere anahtar sorar, tarayicida saklar
# Anahtar: https://aistudio.google.com/apikey
# Model adlari zamanla degisiyor (eskiler emekli oluyor). Asagidaki varsayilan
# calismazsa rapor sayfasi hesabinda MEVCUT modelleri sorup uygun olani kendi
# secer ve tarayicida hatirlar; elle degistirmen gerekmez.
GEMINI_MODEL    = "gemini-3.5-flash"
GEMINI_KEY_FILE = os.path.join(BASE_DIR, "gemini_key.txt")
AI_PANEL        = True          # False yaparsan rapora sohbet paneli eklenmez


def _gemini_key():
    """gemini_key.txt icinden anahtari ayiklar.

    Notepad'in ekledigi BOM, tirnak, bosluk, satir sonu ve 'key=' gibi onekler
    temizlenir; bunlar 'API key not valid' hatasinin en sik sebebidir.
    """
    try:
        if not os.path.exists(GEMINI_KEY_FILE):
            return ""
        with open(GEMINI_KEY_FILE, encoding="utf-8-sig") as f:
            ham = f.read()
    except Exception as e:
        print("[ai] anahtar dosyasi okunamadi:", e)
        return ""
    import re as _re
    m = _re.search(r"AIza[0-9A-Za-z_\-]{30,}", ham)
    if m:
        return m.group(0)
    k = ham.strip().strip('"\'').replace("\ufeff", "").strip()
    if k and not k.lower().startswith(("#", "key", "api")):
        print("[ai] UYARI: gemini_key.txt icindeki deger 'AIza' ile baslamiyor — "
              "anahtar gecersiz olabilir.")
        return k
    if ham.strip():
        print("[ai] UYARI: gemini_key.txt icinde gecerli bir anahtar bulunamadi.")
    return ""


def _rapor_verisi(gun, o, saat_ozet, vard, durus, rows, gunluk, va=False):
    """Yapay zekaya verilecek ozet veri (rapordaki tum sayilar)."""
    r1 = lambda v: round(float(v), 1)

    def dagilim(cevrimler):
        v = sorted(cevrimler)
        if not v:
            return {}
        med = statistics.median(v)
        if len(v) >= 4:
            q = statistics.quantiles(v, n=4)
            q1, q3 = q[0], q[2]
        else:
            q1, q3 = v[0], v[-1]
        return {"min": r1(v[0]), "p25": r1(q1), "medyan": r1(med),
                "p75": r1(q3), "max": r1(v[-1])}

    def blok(z):
        d = {"adet": z["adet"], "ort_cevrim_sn": r1(z["ort_cevrim"]),
             "cmin": r1(z["cmin"]), "parca_saat": r1(z["parca_saat"]),
             "islem_sn": r1(z["ort_islem_span"]),
             "bos_ilk_isleme_kadar_sn": r1(z["ort_nva_bas"]),
             "bos_islemler_arasi_sn": r1(z["ort_nva_ara"]),
             "bos_son_islemden_sonra_sn": r1(z["ort_nva_son"]),
             "islem_yuzde": r1(z["islem_yuzde"]), "bos_yuzde": r1(z["bos_yuzde"]),
             "en_iyi_sn": r1(z["en_iyi"]), "en_kotu_sn": r1(z["en_kotu"])}
        if va:
            d["va_sn"] = r1(z["ort_va_span"])
            d["diger_islem_sn"] = r1(z["ort_nnva_span"])
            d["va_yuzde"] = r1(z["va_yuzde"])
        d.update(dagilim(z["cevrimler"]))
        return d

    return {
        "gun": gun,
        "vardiyalar": [{"ad": a, "baslangic": f"{b//60:02d}:{b%60:02d}",
                        "bitis": f"{s//60:02d}:{s%60:02d}"} for a, b, s in SHIFTS],
        "tanimlar": {
            "cevrim": "Bir parca postadan ciktigi andan bir sonraki parca ciktigi ana kadar. "
                      "Motor gelirken yapilan islemler de bu sureye dahildir.",
            "islem": "Izlenen islem etiketlerinden en az birinin goruldugu sure: "
                     + ", ".join(pretty(x) for x in IS_CLASSES),
            **({"VA": "Katma degerli islem = motora parca takan islemler: "
                      + ", ".join(pretty(x) for x in sorted(VA_OPERATIONS)),
                "diger_islem": "Gerekli ama katma degersiz islemler (NVA): "
                               + ", ".join(pretty(x) for x in NNVA_OPERATIONS)}
               if va else {}),
            "bos_ilk_isleme_kadar": "Cevrim basindan ilk islem gorulene kadar gecen bos sure",
            "bos_islemler_arasi": "Islemler arasindaki bos bosluklar",
            "bos_son_islemden_sonra": "Son islemden cevrim sonuna kadar bos sure",
            "cmin": "Santidakika; 1 dakika = 100 cmin (cevrim_sn / 0,6). "
                    "Rapor ekraninda TUM sureler cmin olarak gosterilir; JSON'da "
                    "saniye cinsinden verilenleri 0,6'ya bolerek cmin'e cevir ve "
                    "yanitlarinda cmin kullan (gerekirse parantezde saniye ver).",
            "eksik_adim": "Bir cevrim kapandiginda isaretlenmemis kalan QA adimlari. "
                          "Eksik adim gercek atlanan islem olabilecegi gibi model "
                          "tespit zayifligindan da kaynaklanabilir.",
            "durus": f"{CEVRIM_MAX_SN:.0f} sn'den uzun cevrimler durus sayilir ve "
                     "ortalamalara girmez.",
            "olcum": "Sureler 'iki etiket arasi' yontemiyle olculur: bir islemin ilk "
                     "goruldugu andan son goruldugu ana kadar.",
            "eksik_klipleri": "Eksik kapanan her cevrim icin son "
                              f"{KLIP_SANIYE//60} dakikalik video kaydedilir; "
                              "dosya adi saat ve eksik adimi icerir. Kullanici "
                              "bunlari izleyerek operatorun ne yaptigini gorebilir."},
        "genel": blok(o),
        "vardiya_ozeti": [dict(vardiya=ad, **blok(z)) for ad, z in vard],
        "saatlik": [dict(saat=s + ":00", **blok(z)) for s, z in saat_ozet],
        "adim_sureleri_sn": {pretty(k): r1(v) for k, v in
                             sorted(o["ort_adim"].items(), key=lambda x: -x[1])},
        "en_cok_eksik_kalan_adimlar": dict(sorted(o["eksik_sayac"].items(),
                                                  key=lambda x: -x[1])[:6]),
        "duruslar": [{"baslangic": r.get("baslangic"), "bitis": r.get("bitis"),
                      "sure_dk": r1(_f(r.get("cevrim_sn")) / 60.0)} for r in durus][:15],
        "son_cevrimler": [{"bitis": r.get("bitis"), "cevrim_sn": r1(_f(r.get("cevrim_sn"))),
                           "islem_sn": r1(_f(r.get("va_span_sn"))
                                          + _f(r.get("nnva_span_sn"))),
                           "bos_sn": r1(_f(r.get("nva_bas_sn")) + _f(r.get("nva_ara_sn"))
                                        + _f(r.get("nva_son_sn")))}
                          for r in rows[-12:]],
        "eksik_klipleri": (gun_klipleri(gun)[:20] if KLIP_AKTIF else []),
        "gunluk_trend": [dict(gun=g, **{k: blok(z)[k] for k in
                                        ("adet", "ort_cevrim_sn", "islem_yuzde")})
                         for g, z in gunluk],
    }


_AI_SYS = """Sen AI-LINER sisteminin uretim analistisin; yalin uretim (lean manufacturing) uzmanisin ve bir motor montaj \
postasinin gunluk uretim raporunu yorumluyorsun. Rapordaki TUM hesaplanmis veri \
sana JSON olarak verildi.

KURALLAR:
- Sadece verideki sayilara dayan. Veride olmayan bir sey uydurma; bilmiyorsan "veride yok" de.
- Yanitlarin kisa ve somut olsun: en fazla 150-200 kelime. Once bulguyu sayisiyla \
soyle, sonra ne yapilmasi gerektigini. Genel gecer nasihat verme. Cevabini mutlaka \
tamamla; yarim cumleyle bitirme.
- Turkce yaz. Baslik yerine kisa paragraflar ve gerektiginde madde isaretleri kullan.
- Bir sorun gordugunde once nedenini veriden gostermeye calis (hangi saat, hangi vardiya, \
hangi bos sure bileseni, hangi adim).
- Kaybi bos surede ara: "ilk isleme kadar", "islemler arasi" ve "son islemden sonra" \
bloklarini ayri ayri degerlendir; en buyugu once ele alinir.
- Veride va_sn / diger_islem_sn varsa katma deger ayrimi aciktir: "diger islemler" \
gerekli ama katma degersiz istir (kisaltilir, yok edilmez); bu alanlar yoksa katma \
deger analizi kapalidir, VA/NVA'dan hic soz etme.
- Cok kisa islem suresi veya sik eksik adim gordugunde, bunun kamera/model tespit \
zayifligindan kaynaklanabilecegini de bir ihtimal olarak belirt.
- Onerileri uygulanabilir ve postaya ozel yaz (or. "16:00-19:00 arasi islemler arasi bos \
sure 30+ sn: aparat yerlesimi ve parca besleme kontrol edilmeli").

VERI (JSON):
"""


def _ai_panel(veri, key):
    """Rapor sayfasina gomulen sohbet paneli (HTML+JS)."""
    js = r"""
<div id="ai" class="ai">
  <div class="ai-bar" onclick="aiAc()">
    <span class="ai-nokta"></span><b>AI-LINER analisti</b>
    <span class="ai-min" id="aiMin">?</span>
  </div>
  <div class="ai-govde" id="aiGovde">
    <div class="ai-msj" id="aiMsj">
      <div class="ai-bot ai-bal">Merhaba. Bu raporun tum verisi elimde — sormak
      istedigini yaz ya da asagidaki hazir sorulardan birine bas.</div>
    </div>
    <div class="ai-hazir">
      <button onclick="aiHazir(this)">Bu gunu ozetle</button>
      <button onclick="aiHazir(this)">En buyuk kayip nerede?</button>
      <button onclick="aiHazir(this)">Duruslari degerlendir</button>
      <button onclick="aiHazir(this)">Vardiyalari karsilastir</button>
      <button onclick="aiHazir(this)">En cok eksik kalan adimlar</button>
      <button onclick="aiHazir(this)">3 somut iyilestirme onerisi ver</button>
    </div>
    <div class="ai-giris">
      <textarea id="aiSoru" rows="1" placeholder="Soru yaz, Enter ile gonder..."
        onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();aiGonder();}"></textarea>
      <button class="ai-gonder" onclick="aiGonder()">Gonder</button>
    </div>
    <div class="ai-alt">
      <span id="aiDurum"></span>
      <span><a href="#" onclick="aiAnahtar();return false;">API anahtari</a>
      &nbsp;·&nbsp;<a href="#" onclick="aiTest();return false;">anahtari dene</a>
      &nbsp;·&nbsp;<a href="#" onclick="aiModelDegistir();return false;">model</a></span>
    </div>
  </div>
</div>
<script>
const AI_VERI = __VERI__;
const AI_SYS  = __SYS__ + JSON.stringify(AI_VERI);
const AI_GOMULU = "__KEY__";
const AI_MODEL_VARSAYILAN = "__MODEL__";
const AI_URL = "https://generativelanguage.googleapis.com/v1beta/models";
let aiGecmis = [], aiMesgul = false, aiSonIstek = 0;
const AI_MIN_ARA = 4000;        // iki istek arasi en az sure (kota korumasi)

function aiUyu(ms){ return new Promise(r => setTimeout(r, ms)); }

function aiBeklemeSuresi(j){
  // 429 yanitinda Google onerilen bekleme suresini dondurur (RetryInfo)
  try {
    for (const d of (j.error.details || [])){
      const t = d["@type"] || "";
      if (t.includes("RetryInfo") && d.retryDelay)
        return Math.min(120, Math.ceil(parseFloat(d.retryDelay)) || 0);
    }
  } catch(e){}
  return 0;
}
async function aiGeriSayim(kutu, sn, metin){
  for (let k = sn; k > 0; k--){
    kutu.innerHTML = aiBicim(metin + " **" + k + " sn**...");
    aiDurum("Kota bekleniyor: " + k + " sn");
    await aiUyu(1000);
  }
  aiDurum("");
}
async function aiHafifModel(){
  // kota model basina ayri: daha hafif bir modele gecmek cogu zaman isi kurtarir
  try {
    const l = await aiModelListesi();
    const hafif = l.filter(n => n.includes("lite"))
                   .sort((a, b) => aiModelPuan(b) - aiModelPuan(a))[0];
    if (hafif){
      try { localStorage.setItem("gemini_model", hafif); } catch(e){}
      return hafif;
    }
  } catch(e){}
  return "";
}

function aiModel(){
  try { return localStorage.getItem("gemini_model") || AI_MODEL_VARSAYILAN; }
  catch(e){ return AI_MODEL_VARSAYILAN; }
}
async function aiModelListesi(){
  const y = await fetch(AI_URL + "?key=" + encodeURIComponent(aiKey()));
  const j = await y.json().catch(()=>({}));
  if (!y.ok) throw new Error("HTTP " + y.status + ": "
      + ((j.error && j.error.message) || ""));
  return (j.models || [])
    .filter(m => (m.supportedGenerationMethods || []).includes("generateContent"))
    .map(m => m.name.replace("models/", ""))
    .filter(n => !/embedding|aqa|image|tts|live|transcribe|vision|guard/.test(n));
}
function aiModelPuan(n){
  let p = 0;
  if (n.includes("flash")) p += 100;      // hizli ve ucuz olani tercih et
  if (n.includes("lite")) p -= 5;
  if (/preview|exp|experimental/.test(n)) p -= 200;
  const m = n.match(/(\d+(?:\.\d+)?)/);
  if (m) p += parseFloat(m[1]) * 10;      // yeni surum daha iyi
  return p;
}
async function aiModelSec(sessiz){
  const uygun = await aiModelListesi();
  uygun.sort((a, b) => aiModelPuan(b) - aiModelPuan(a));
  const sec = uygun[0];
  if (!sec) return "";
  try { localStorage.setItem("gemini_model", sec); } catch(e){}
  if (!sessiz) aiEkle("bot", "Model **" + sec + "** olarak ayarlandi.");
  return sec;
}
async function aiModelDegistir(){
  try {
    const l = await aiModelListesi();
    l.sort((a, b) => aiModelPuan(b) - aiModelPuan(a));
    const m = prompt("Kullanilacak model (hesabinda mevcut olanlar):\n\n"
                     + l.join("\n"), aiModel());
    if (m === null) return;
    try { localStorage.setItem("gemini_model", m.trim()); } catch(e){}
    aiDurum("Model: " + m.trim());
  } catch(e){ aiEkle("bot", "Model listesi alinamadi: " + e.message + aiAgIpucu()); }
}

function aiTemizle(k){
  if (!k) return "";
  k = k.replace(/[\u200b\ufeff]/g, "").trim().replace(/^["']|["']$/g, "").trim();
  const m = k.match(/AIza[0-9A-Za-z_\-]{30,}/);   // etraftaki yaziyi at
  return m ? m[0] : k;
}
function aiKey(){
  // Sayfaya girilen anahtar, dosyadan gomulene gore ONCELIKLI:
  // gomulu anahtar bozuksa kullanici sayfadan duzeltebilsin.
  let k = "";
  try { k = aiTemizle(localStorage.getItem("gemini_key") || ""); } catch(e){}
  return k || aiTemizle(AI_GOMULU);
}
function aiKaynak(){
  let k = "";
  try { k = aiTemizle(localStorage.getItem("gemini_key") || ""); } catch(e){}
  return k ? "sayfada kayitli" : (aiTemizle(AI_GOMULU) ? "gemini_key.txt" : "yok");
}
function aiAnahtar(){
  const eski = (()=>{try{return localStorage.getItem("gemini_key")||"";}catch(e){return "";}})();
  const m = prompt("Gemini API anahtarini yapistir (aistudio.google.com/apikey).\n"
                 + "Bosaltip OK dersen sayfadaki anahtar silinir.", eski);
  if (m === null) return;
  const k = aiTemizle(m);
  try {
    if (k) localStorage.setItem("gemini_key", k);
    else localStorage.removeItem("gemini_key");
  } catch(e){ aiDurum("Tarayici anahtari saklayamadi."); return; }
  if (k && !/^AIza[0-9A-Za-z_\-]{30,}$/.test(k)){
    aiEkle("bot", "**Dikkat:** girdigin deger bir Gemini anahtarina benzemiyor. "
      + "Anahtar `AIza` ile baslar ve ~39 karakterdir. Google Cloud'daki proje "
      + "numarasi veya OAuth istemci kimligi degil, **API key** olmali.");
  }
  aiDurum(k ? "Anahtar kaydedildi (" + k.slice(0,6) + "…" + k.slice(-4) + ")." : "Anahtar silindi.");
  if (k) aiTest();
}
async function aiTest(){
  const key = aiKey();
  if (!key){ aiEkle("bot", "Kayitli anahtar yok."); return; }
  aiDurum("Anahtar deneniyor...");
  try {
    const y = await fetch("https://generativelanguage.googleapis.com/v1beta/models?key="
                          + encodeURIComponent(key));
    const j = await y.json().catch(()=>({}));
    if (y.ok){
      const uygun = (j.models || [])
        .filter(m => (m.supportedGenerationMethods || []).includes("generateContent"))
        .map(m => m.name.replace("models/", ""))
        .filter(n => !/embedding|aqa|image|tts|live|transcribe|vision|guard/.test(n));
      uygun.sort((a, b) => aiModelPuan(b) - aiModelPuan(a));
      let mevcut = aiModel(), notu = "";
      if (!uygun.includes(mevcut) && uygun.length){
        mevcut = uygun[0];
        try { localStorage.setItem("gemini_model", mevcut); } catch(e){}
        notu = " Onceki model kullanilamiyordu, otomatik olarak buna gecildi.";
      }
      aiEkle("bot", "Anahtar **gecerli** (" + aiKaynak() + "). Sohbet icin uygun "
        + uygun.length + " model var. Kullanilacak model: **" + mevcut + "**." + notu
        + (uygun.length ? "\n\nIlk secenekler: " + uygun.slice(0, 5).join(", ") : ""));
      aiDurum("Model: " + mevcut);
    } else {
      aiEkle("bot", "Anahtar **reddedildi** (HTTP " + y.status + "). "
        + aiHataIpucu(y.status, (j.error && j.error.message) || ""));
      aiDurum("");
    }
  } catch(e){
    aiEkle("bot", "Baglanti kurulamadi: " + e.message + aiAgIpucu());
    aiDurum("");
  }
}
function aiAgIpucu(){
  return location.protocol === "file:"
    ? " Sayfa dosyadan acildigi icin tarayici istegi engellemis olabilir: "
      + "uygulamadaki **Rapor + AI (yerel sunucu)** butonuyla ac."
    : " Internet baglantisini / vekil sunucuyu kontrol et.";
}
function aiHataIpucu(kod, mesaj){
  const m = (mesaj || "").toLowerCase();
  if (kod === 400 && m.includes("api key not valid"))
    return "Anahtar Google tarafindan taninmadi. Sik sebepler: (1) kopyalarken "
      + "basina/sonuna fazladan karakter gelmis, (2) anahtar silinmis ya da baska "
      + "bir projeye ait, (3) anahtarin olusturuldugu projede **Generative Language "
      + "API** etkin degil. En hizli cozum: aistudio.google.com/apikey adresinden "
      + "YENI bir anahtar olusturup **API anahtari** baglantisindan yapistir.";
  if (kod === 403 && (m.includes("referer") || m.includes("blocked")))
    return "Anahtar kisitlanmis (HTTP referrer / IP kisiti). Google Cloud > "
      + "Credentials icinde bu anahtarin kisitlarini kaldir veya localhost'u ekle.";
  if (kod === 403)
    return "Erisim reddedildi: anahtarin projesinde Generative Language API etkin mi?";
  if (kod === 429)
    return "Kota siniri. Ucretsiz katmanda dakikada ve gunde sinirli sayida istek "
      + "hakkin var; art arda soru sorunca dakikalik sinir hemen dolar.\n\n"
      + "- Bir dakika bekleyip tekrar dene (panel bunu otomatik de yapar)\n"
      + "- Alttaki **model** baglantisindan `-lite` biten bir modele gec "
      + "(kota model basina ayridir)\n"
      + "- Gunluk hak bittiyse ertesi gunu bekle ya da AI Studio'da faturalandirmayi "
      + "acip ucretli katmana gec\n\n"
      + "Guncel limitler: ai.google.dev/gemini-api/docs/rate-limits";
  if (kod === 404)
    return "Model bulunamadi (`" + aiModel() + "`) — bu isim emekli olmus olabilir. "
      + "Alttaki **model** baglantisindan hesabindaki modellerden birini sec "
      + "(ya da **anahtari dene**, uygun olani otomatik ayarlar).";
  return mesaj || "Ayrinti yok.";
}
function aiAc(){
  const g = document.getElementById("aiGovde");
  const k = g.style.display === "block";
  g.style.display = k ? "none" : "block";
  document.getElementById("aiMin").textContent = k ? "?" : "_";
  if (!k) document.getElementById("aiSoru").focus();
}
function aiDurum(t){ document.getElementById("aiDurum").textContent = t || ""; }
function aiKacir(s){ return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function aiBicim(t){
  let h = aiKacir(t);
  h = h.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
  h = h.replace(/^\s*[-*•]\s+(.*)$/gm, "<li>$1</li>");
  h = h.replace(/(<li>[\s\S]*?<\/li>)/g, "<ul>$1</ul>");
  return h.replace(/\n{2,}/g, "<br><br>").replace(/\n/g, "<br>");
}
function aiEkle(rol, metin){
  const d = document.createElement("div");
  d.className = "ai-bal " + (rol === "user" ? "ai-ben" : "ai-bot");
  d.innerHTML = rol === "user" ? aiKacir(metin) : aiBicim(metin);
  const k = document.getElementById("aiMsj");
  k.appendChild(d); k.scrollTop = k.scrollHeight;
  return d;
}
function aiHazir(b){
  if (aiMesgul){ aiDurum("Once mevcut yanit bitsin."); return; }
  document.getElementById("aiSoru").value = b.textContent;
  aiGonder();
}
async function aiGonder(){
  if (aiMesgul) return;
  const el = document.getElementById("aiSoru");
  const soru = el.value.trim();
  if (!soru) return;
  const key = aiKey();
  if (!key){
    aiEkle("bot", "Once bir Gemini API anahtari gerekli. Alttaki **API anahtari** "
      + "baglantisina basip yapistir.\n\naistudio.google.com/apikey adresinde "
      + "**Create API key** de; olusan `AIza...` ile baslayan metni oldugu gibi "
      + "kopyala. Kaydettikten sonra anahtar otomatik denenir.");
    return;
  }
  el.value = ""; aiEkle("user", soru);
  aiMesgul = true; aiDurum("Dusunuyor...");
  document.querySelectorAll(".ai-hazir button").forEach(b => b.disabled = true);
  const bekle = aiEkle("bot", "…");
  try {
    // NOT: yeni Gemini modelleri varsayilan olarak "dusunur" ve bu dusunme
    // token'lari cikti butcesinden yenir -> yanit cumle ortasinda kesilir.
    // Bu yuzden butce buyutuldu ve dusunme kapatildi (desteklemeyen model
    // 400 dondururse istek dusunme ayari olmadan tekrarlanir).
    const govde = {
      system_instruction: { parts: [{ text: AI_SYS }] },
      contents: aiGecmis.concat([{ role: "user", parts: [{ text: soru }] }]),
      generationConfig: { temperature: 0.4, maxOutputTokens: 8192,
                          thinkingConfig: { thinkingBudget: 0 } }
    };
    const istek = (m, g) => fetch(AI_URL + "/" + m + ":generateContent?key="
        + encodeURIComponent(key),
        { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(g || govde) });
    // arka arkaya hizli tiklamalarda kotayi bosa harcama
    const gecen = Date.now() - aiSonIstek;
    if (aiSonIstek && gecen < AI_MIN_ARA){
      const kalan = Math.ceil((AI_MIN_ARA - gecen) / 1000);
      bekle.innerHTML = aiBicim("Sirada... **" + kalan + " sn**");
      await aiUyu(AI_MIN_ARA - gecen);
    }
    aiSonIstek = Date.now();

    let model = aiModel();
    let y = await istek(model);

    // ── 429: kota. Onerilen sureyi bekleyip bir kez tekrar dene ──
    if (y.status === 429){
      let j429 = {}; try { j429 = await y.clone().json(); } catch(e){}
      const sn = aiBeklemeSuresi(j429) || 25;
      await aiGeriSayim(bekle, sn, "Kota siniri (429). Otomatik tekrar denenecek:");
      aiSonIstek = Date.now();
      y = await istek(model);
      if (y.status === 429){
        const hafif = await aiHafifModel();
        if (hafif && hafif !== model){
          bekle.innerHTML = aiBicim("Kota hala dolu — daha hafif **" + hafif
                                    + "** modeline geciliyor...");
          model = hafif; aiSonIstek = Date.now();
          y = await istek(model);
        }
      }
    }
    if (y.status === 404){
      // model emekli olmus / hesapta yok -> mevcutlardan uygun olani sec, tekrar dene
      aiDurum("Model degisiyor...");
      let yeni = "";
      try { yeni = await aiModelSec(true); } catch(e){}
      if (yeni && yeni !== model){
        bekle.innerHTML = aiBicim("`" + model + "` kullanilamiyor, **" + yeni
                                  + "** modeline gecildi. Yanit hazirlaniyor...");
        model = yeni;
        y = await istek(model);
      }
    }
    if (y.status === 400){
      let msj = ""; try { msj = (await y.clone().json()).error.message || ""; } catch(e){}
      if (/thinking|thinkingConfig|thinking_budget/i.test(msj)){
        const g2 = JSON.parse(JSON.stringify(govde));
        delete g2.generationConfig.thinkingConfig;   // bu model ayari desteklemiyor
        y = await istek(model, g2);
      }
    }
    if (!y.ok){
      let ek = ""; try { ek = (await y.json()).error.message; } catch(e){}
      bekle.innerHTML = aiBicim("**Istek reddedildi (HTTP " + y.status + ").** "
        + aiHataIpucu(y.status, ek));
      aiDurum(""); aiMesgul = false; return;
    }
    const j = await y.json();
    const c = j.candidates && j.candidates[0];
    const parcalar = (c && c.content && c.content.parts) || [];
    let metin = parcalar.filter(p => !p.thought).map(p => p.text || "").join("").trim();
    const neden = c && c.finishReason;
    if (!metin){
      metin = neden === "MAX_TOKENS"
        ? "Yanit uretilemedi: model dusunmeye ayrilan butceyi tuketti. "
          + "Soruyu biraz daha dar sor ya da alttaki **model** baglantisindan "
          + "baska bir model sec."
        : "(bos yanit" + (neden ? " — " + neden : "") + ")";
    } else if (neden === "MAX_TOKENS"){
      metin += "\n\n_(yanit uzunluk sinirina takildi — devami icin 'devam et' yaz)_";
    } else if (neden === "SAFETY" || neden === "RECITATION"){
      metin += "\n\n_(yanit " + neden + " nedeniyle erken bitti)_";
    }
    bekle.innerHTML = aiBicim(metin);
    aiGecmis.push({ role: "user", parts: [{ text: soru }] });
    aiGecmis.push({ role: "model", parts: [{ text: metin }] });
    if (aiGecmis.length > 6) aiGecmis = aiGecmis.slice(-6);   // kota icin kisa tut
    aiDurum("");
  } catch(e){
    bekle.innerHTML = aiBicim("Hata: " + e.message
      + (String(e).includes("Failed to fetch") ? aiAgIpucu() : ""));
    aiDurum("");
  }
  aiMesgul = false;
  document.querySelectorAll(".ai-hazir button").forEach(b => b.disabled = false);
  document.getElementById("aiMsj").scrollTop = 1e9;
}
</script>
"""
    return (js.replace("__VERI__", json.dumps(veri, ensure_ascii=False))
              .replace("__SYS__", json.dumps(_AI_SYS, ensure_ascii=False))
              .replace("__KEY__", (key or "").replace('"', ""))
              .replace("__MODEL__", GEMINI_MODEL))


_AI_CSS = """
.ai{position:fixed;right:18px;bottom:18px;width:380px;max-width:calc(100vw - 36px);
z-index:50;font-size:13px}
.ai-bar{background:var(--s1);color:#fff;border-radius:10px;padding:9px 13px;cursor:pointer;
display:flex;align-items:center;gap:8px;box-shadow:0 4px 16px rgba(0,0,0,.22)}
.ai-bar b{flex:1;font-weight:600}
.ai-min{opacity:.8}
.ai-nokta{width:8px;height:8px;border-radius:50%;background:#8ff0b0;display:inline-block}
.ai-govde{display:none;background:var(--surface);border:1px solid var(--line);
border-radius:10px;margin-top:8px;box-shadow:0 8px 30px rgba(0,0,0,.22);overflow:hidden}
.ai-msj{max-height:44vh;overflow-y:auto;padding:12px}
.ai-bal{padding:8px 11px;border-radius:9px;margin-bottom:8px;line-height:1.55}
.ai-bot{background:rgba(127,127,127,.10);color:var(--ink)}
.ai-ben{background:var(--s1);color:#fff;margin-left:36px}
.ai-bal ul{margin:6px 0 0;padding-left:18px}
.ai-hazir{display:flex;flex-wrap:wrap;gap:5px;padding:0 12px 10px}
.ai-hazir button{font:inherit;font-size:11.5px;padding:4px 9px;border-radius:20px;
border:1px solid var(--line);background:transparent;color:var(--ink2);cursor:pointer}
.ai-hazir button:hover{background:rgba(127,127,127,.12);color:var(--ink)}
.ai-hazir button:disabled{opacity:.45;cursor:default}
.ai-giris{display:flex;gap:6px;padding:10px 12px;border-top:1px solid var(--line)}
.ai-giris textarea{flex:1;resize:none;font:inherit;padding:7px 9px;border-radius:7px;
border:1px solid var(--line);background:var(--plane);color:var(--ink)}
.ai-gonder{font:inherit;padding:7px 13px;border:0;border-radius:7px;background:var(--s1);
color:#fff;cursor:pointer}
.ai-alt{display:flex;justify-content:space-between;padding:0 12px 10px;font-size:11px;
color:var(--ink3)}
.ai-alt a{color:var(--ink3)}
@media print{.ai{display:none}}
"""


def cevrim_satiri(cyc, t_end, sonuc, ok, total, eksik):
    """Biten bir cevrimi CSV satirina cevirir.

    Cevrim, onceki parca postadan ciktigi anda baslar: motor gelirken yapilan
    islemler de sayilir. Olculen iki buyukluk var: ISLEM (bir islem etiketi
    goruldugu sure) ve BOS (hicbir islem gorulmeyen sure).

    CSV her zaman VA/NVA ayrimini da tutar (va_span_sn = motora parca takan
    islemler, nnva_span_sn = diger izlenen islemler). Bu ayrim SADECE HTML
    raporunda ve yalnizca VA_RAPOR acikken gosterilir; canli panellerde hic
    kullanilmaz. Ikisinin TOPLAMI toplam islem suresidir.
    """
    cevrim  = max(0.0, cyc["dur"])
    bekleme = max(0.0, min(cyc.get("arm_dur") or 0.0, cevrim))  # motor yerlesene kadar
    posta   = max(0.0, cevrim - bekleme)
    va      = max(0.0, min(cyc["va"], cevrim))
    nva_p   = max(0.0, posta - min(va, posta))

    # ── "iki etiket arasi" olcum: islem araliklari ──
    ivs    = _merge_spans(cyc["spans"])                       # her turlu islem
    ivs_va = _merge_spans({k: v for k, v in cyc["spans"].items()
                           if k in VA_OPERATIONS})            # motora takilanlar
    aktif   = sum(b - a for a, b in ivs)
    va_span = sum(b - a for a, b in ivs_va)
    nnva_span = max(0.0, aktif - va_span)                     # diger islemler
    if ivs:
        ilk, son = ivs[0][0], ivs[-1][1]
        nva_bas = max(0.0, ilk - cyc["t0"])
        nva_son = max(0.0, t_end - son)
        nva_ara = max(0.0, (son - ilk) - aktif)
    else:
        nva_ara = nva_son = 0.0
        nva_bas = cevrim

    d0, d1 = datetime.fromtimestamp(cyc["t0"]), datetime.fromtimestamp(t_end)
    row = {
        "tarih": d0.strftime("%Y-%m-%d"), "saat": d0.strftime("%H"),
        "vardiya": vardiya_adi(d0),
        "baslangic": d0.strftime("%H:%M:%S"), "bitis": d1.strftime("%H:%M:%S"),
        "cevrim_sn": _num(cevrim), "cmin": _num(cevrim / SEC_PER_CMIN, 1),
        "parca_saat": _num(3600.0 / cevrim if cevrim > 0 else 0.0, 1),
        "bekleme_sn": _num(bekleme), "posta_sn": _num(posta),
        "va_sn": _num(va), "nva_posta_sn": _num(nva_p),
        "nva_toplam_sn": _num(max(0.0, cevrim - va)),
        "va_yuzde": _num(100.0 * va_span / cevrim if cevrim > 0 else 0.0, 1),
        "va_span_sn": _num(va_span), "nnva_span_sn": _num(nnva_span),
        "nva_bas_sn": _num(nva_bas),
        "nva_ara_sn": _num(nva_ara), "nva_son_sn": _num(nva_son),
        "sonuc": sonuc or "", "ok_adim": ok, "toplam_adim": total,
        "eksik": ";".join(pretty(s) for s in (eksik or [])),
    }
    for s, c in STEP_COL.items():
        row[f"sure_{c}"] = _num(cyc["step_sec"].get(s, 0.0))
        row[f"bitis_{c}"] = _num(cyc["done_at"][s]) if s in cyc["done_at"] else ""
    return row


def _boxes(gruplar, birim="sn"):
    """Dagilim: her grup icin min-max bicak, %25-%75 kutu, medyan cizgisi."""
    gruplar = [(e, sorted(v)) for e, v in gruplar if v]
    if not gruplar:
        return '<p class="alt">Veri yok.</p>'
    enb = max(max(v) for _, v in gruplar) or 1.0
    cols, xs = [], []
    for etiket, v in gruplar:
        mn, mx = v[0], v[-1]
        med = statistics.median(v)
        if len(v) >= 4:
            q = statistics.quantiles(v, n=4)
            q1, q3 = q[0], q[2]
        else:
            q1, q3 = mn, mx
        p = lambda x: 100.0 * x / enb
        ipucu = (f"{etiket} · n={len(v)} · min {nf(mn)} / %25 {nf(q1)} / "
                 f"medyan {nf(med)} / %75 {nf(q3)} / max {nf(mx)} {birim}")
        cols.append(
            f'<div class="c"><div class="tot">{nf(med)}</div>'
            f'<div class="box" title="{ipucu}">'
            f'<div class="wk" style="bottom:{p(mn):.2f}%;height:{p(mx-mn):.2f}%"></div>'
            f'<div class="bx" style="bottom:{p(q1):.2f}%;height:{max(2.0,p(q3-q1)):.2f}%"></div>'
            f'<div class="md" style="bottom:{p(med):.2f}%"></div>'
            f'<div class="tk" style="bottom:{p(mn):.2f}%"></div>'
            f'<div class="tk" style="bottom:{p(mx):.2f}%"></div>'
            f'</div></div>')
        xs.append(f"<div>{etiket}</div>")
    return ('<div class="cols">' + "".join(cols) + '</div><div class="xax">'
            + "".join(xs) + "</div>")


def rapor_html(logger, gun=None, gun_sayisi=7, va=None):
    """va=True ise raporda KATMA DEGER (VA/NVA) ayrimi gosterilir.
    None verilirse VA_RAPOR ayari kullanilir. Canli panellerde VA hic yok."""
    va = VA_RAPOR if va is None else bool(va)
    gun = gun or datetime.now().strftime("%Y-%m-%d")
    ham = logger.load(gun)
    rows, durus = gecerli_satirlar(ham)
    vardiya_disi = False
    if not rows and ham:
        # vardiya saatlerinde kayit yok (gece testi vb.) -> hepsini goster
        rows, durus = gecerli_satirlar(ham, vardiya_zorunlu=False)
        vardiya_disi = bool(rows)
    o = _ozetle(rows)
    vardiya_listesi = [ad for ad, _, _ in SHIFTS]

    h = [_rapor_basligi(gun),
         f'<p class="alt">Rapor gunu: <b>{gun}</b> · Olusturma: {datetime.now().strftime("%d.%m.%Y %H:%M")} · '
         f'kaynak: cevrimler_{gun}.csv · 1 dakika = 100 cmin · '
         f'vardiyalar: ' + " / ".join(
             f"{ad} {b//60:02d}:{b%60:02d}-{s//60:02d}:{s%60:02d}"
             for ad, b, s in SHIFTS)
         + f' · {nf(CEVRIM_MAX_SN,0)} sn ustu cevrimler DURUS sayildi</p>']

    if not rows:
        h.append('<div class="card"><p class="alt" style="margin:0">'
                 f'Bu gun icin kayitli cevrim yok ({len(ham)} ham kayit okundu).'
                 '</p></div>')
        return _sayfa(gun, "".join(h))

    if vardiya_disi:
        h.append('<div class="card" style="border-color:var(--s4);margin-bottom:16px">'
                 f'<p class="alt" style="margin:0">⚠ Bu gun vardiya saatleri icinde '
                 f'({" / ".join(ad for ad, _, _ in SHIFTS)}) hic kayit yok. '
                 f'{len(rows)} adet VARDIYA DISI kayit gosteriliyor — gece testi '
                 'veya mesai disi calisma olabilir.</p></div>')

    durus_sn = sum(_f(r.get("cevrim_sn")) for r in durus)
    eski_n = sum(1 for r in rows if r.get("_eski"))
    if eski_n:
        h.append('<div class="card" style="border-color:var(--s2);margin-bottom:16px">'
                 f'<p class="alt" style="margin:0">⚠ {eski_n} cevrim eski surumle '
                 'kaydedilmis: o kayitlarda "iki etiket arasi" olcumu yok. Islem '
                 'suresi eski kare-sayma degerinden, "ilk isleme kadar" eski bekleme '
                 'suresinden turetildi; "islemler arasi" ve "son islemden sonra" '
                 'ayrimi bu satirlar icin yapilamiyor.</p></div>')

    # ── KPI ──
    h.append('<div class="kpi">')
    h.append(_kpi("Toplam parca", f"{o['adet']}", f"{gun} · vardiya ici"))
    h.append(_kpi("Ort. cevrim", f'{sg(o["ort_cevrim"])}<span class="u">{sb()}</span>',
                  f"medyan {st_(o['medyan'])} · {nf(o['ort_cevrim'])} sn"))
    h.append(_kpi("Hiz", f'{nf(o["parca_saat"],1)}<span class="u">parca/saat</span>',
                  f"{nf(o['parca_dk'],2)} parca/dk"))
    if va:
        h.append(_kpi("Katma deger (VA)", f"%{nf(o['va_yuzde'])}",
                      f"NVA %{nf(o['nva_yuzde'])} · {st_(o['ort_va_span'], 1)}/parca"))
    h.append(_kpi("Durus", f'{len(durus)}<span class="u">kez</span>',
                  f"toplam {nf(durus_sn/60.0)} dk · ortalamalara girmedi"))
    h.append("</div>")

    # ── vardiya ozeti ──
    sira = {ad: i for i, (ad, _b, _s) in enumerate(SHIFTS)}
    adlar = []
    for r in rows:
        ad = satir_vardiyasi(r) or "Vardiya disi"
        if ad not in adlar:
            adlar.append(ad)
    adlar.sort(key=lambda a: sira.get(a, 99))
    vard = [(ad, _ozetle([r for r in rows
                          if (satir_vardiyasi(r) or "Vardiya disi") == ad]))
            for ad in adlar]
    if vard:
        h.append("<h2>Vardiya ozeti</h2><div class=\"card\"><table><thead><tr>"
                 f"<th>Vardiya</th><th>Adet</th><th>Ort. cevrim<br>({sb()})</th>"
                 f"<th>parca/saat</th><th>En iyi<br>({sb()})</th>"
                 f"<th>En kotu<br>({sb()})</th>"
                 + ("<th>VA %</th>" if va else "")
                 + "</tr></thead><tbody>")
        for ad, z in vard:
            h.append(f"<tr><td>{ad}</td><td>{z['adet']}</td>"
                     f"<td>{sg(z['ort_cevrim'])}</td>"
                     f"<td>{nf(z['parca_saat'],1)}</td><td>{sg(z['en_iyi'])}</td>"
                     f"<td>{sg(z['en_kotu'])}</td>"
                     + (f"<td>{nf(z['va_yuzde'],0)}</td>" if va else "")
                     + "</tr>")
        h.append("</tbody></table></div>")

    # ── saatlik ──
    saatler = {}
    for r in rows:
        saatler.setdefault((r.get("saat") or "??"), []).append(r)
    saat_ozet = [(s, _ozetle(v)) for s, v in sorted(saatler.items())]

    # alttan uste: (VA -> diger islem) ya da tek islem blogu -> bos bekleme
    if va:
        seri = [("Katma degerli islem (VA)", "var(--s1)"),
                ("Diger islemler (NVA)", "var(--s2)"),
                ("Bos / bekleme (NVA)", "var(--s3)")]
    else:
        seri = [("Islem suresi", "var(--s1)"),
                ("Bos / bekleme", "var(--s3)")]

    def _bos(z):
        return sd(z["ort_nva_bas"] + z["ort_nva_ara"] + z["ort_nva_son"])

    def _bilesen(z):
        if va:
            return {"Katma degerli islem (VA)": sd(z["ort_va_span"]),
                    "Diger islemler (NVA)": sd(z["ort_nnva_span"]),
                    "Bos / bekleme (NVA)": _bos(z)}
        return {"Islem suresi": sd(z["ort_islem_span"]),
                "Bos / bekleme": _bos(z)}

    h.append(f"<h2>Saatlik cevrim suresi bilesimi — ortalama {sb()}/parca</h2>")
    h.append('<div class="card">')
    h.append('<div class="lgn">' + "".join(
        f'<span><i style="background:{c}"></i>{a}</span>' for a, c in seri) + "</div>")
    h.append(_stacked([(s + ":00", _bilesen(z)) for s, z in saat_ozet], seri, sb(), 0))
    if va:
        h.append('<p class="alt" style="margin:10px 0 0">Katma deger (VA) = motora '
                 'parca takan islemler ('
                 + ", ".join(pretty(x) for x in sorted(VA_OPERATIONS))
                 + '). Diger islemler (' + ", ".join(pretty(x) for x in NNVA_OPERATIONS)
                 + ') gerekli olabilir ama parcaya deger katmaz — NVA sayilir. '
                   'Sureler "iki etiket arasi" yontemiyle olculur.</p></div>')
    else:
        h.append('<p class="alt" style="margin:10px 0 0">Islem suresi = izlenen bir '
                 'islem etiketinin goruldugu sure ('
                 + ", ".join(pretty(x) for x in IS_CLASSES)
                 + '). Geri kalan sure bos/bekleme sayilir. Sureler "iki etiket '
                   'arasi" yontemiyle olculur.</p></div>')

    h.append("<h2>Saatlik uretim adedi</h2>")
    h.append('<div class="card">' + _stacked(
        [(s + ":00", {"Adet": float(z["adet"])}) for s, z in saat_ozet],
        [("Adet", "var(--s1)")], "parca", 0) + "</div>")

    # ── saatlik dagilim ──
    h.append("<h2>Saatlik cevrim suresi dagilimi</h2>")
    h.append('<div class="card">'
             '<div class="lgn"><span><i style="background:var(--s1)"></i>'
             '%25–%75 araligi</span><span>| cizgi: medyan · ince cubuk: min–max</span></div>'
             + _boxes([(s + ":00", [sd(v) for v in z["cevrimler"]])
                        for s, z in saat_ozet], sb())
             + '<p class="alt" style="margin:10px 0 0">Kutu ne kadar kisaysa cevrim '
               'o kadar kararli. Uzun ust bicak: tekil takilmalar.</p></div>')

    # ── saatlik tablo ──
    if va:
        is_bas = f"<th>VA<br>{sb()}</th><th>Diger islem<br>{sb()}</th>"
    else:
        is_bas = f"<th>Islem<br>{sb()}</th>"
    h.append("<h2>Saatlik tablo</h2><div class=\"card\"><table><thead><tr>"
             f"<th>Saat</th><th>Adet</th><th>Ort. cevrim<br>({sb()})</th>"
             f"<th>parca/saat</th>{is_bas}"
             "<th>Bos:<br>ilk isleme kadar</th><th>Bos:<br>islemler arasi</th>"
             "<th>Bos:<br>son islemden sonra</th>"
             + ("<th>VA %</th>" if va else "")
             + "</tr></thead><tbody>")

    def _is_hucre(z, kalin=False):
        a, k = ("<td><b>", "</b></td>") if kalin else ("<td>", "</td>")
        if va:
            return (a + sg(z["ort_va_span"], 1) + k
                    + a + sg(z["ort_nnva_span"], 1) + k)
        return a + sg(z["ort_islem_span"], 1) + k

    for s, z in saat_ozet:
        h.append(f"<tr><td>{s}:00</td><td>{z['adet']}</td><td>{sg(z['ort_cevrim'])}</td>"
                 f"<td>{nf(z['parca_saat'],1)}</td>"
                 + _is_hucre(z)
                 + f"<td>{sg(z['ort_nva_bas'],1)}</td><td>{sg(z['ort_nva_ara'],1)}</td>"
                 + f"<td>{sg(z['ort_nva_son'],1)}</td>"
                 + (f"<td>{nf(z['va_yuzde'],0)}</td>" if va else "")
                 + "</tr>")
    h.append(f"<tr><td><b>GUN</b></td><td><b>{o['adet']}</b></td>"
             f"<td><b>{sg(o['ort_cevrim'])}</b></td>"
             f"<td><b>{nf(o['parca_saat'],1)}</b></td>"
             + _is_hucre(o, True)
             + f"<td><b>{sg(o['ort_nva_bas'],1)}</b></td>"
             + f"<td><b>{sg(o['ort_nva_ara'],1)}</b></td>"
             + f"<td><b>{sg(o['ort_nva_son'],1)}</b></td>"
             + (f"<td><b>{nf(o['va_yuzde'],0)}</b></td>" if va else "")
             + "</tr>")
    h.append("</tbody></table></div>")

    # ── son cevrimler ──
    son = rows[-SON_CEVRIM_N:]
    h.append(f"<h2>Son {len(son)} cevrim — en sagdaki en son parca</h2>")
    gruplar = []
    for i, r in enumerate(son):
        et = r.get("bitis", "")[:5]
        if i == len(son) - 1:
            et = "SON " + et
        r_bos = sd(_f(r.get("nva_bas_sn")) + _f(r.get("nva_ara_sn"))
                   + _f(r.get("nva_son_sn")))
        if va:
            gruplar.append((et, {
                "Katma degerli islem (VA)": sd(_f(r.get("va_span_sn"))),
                "Diger islemler (NVA)": sd(_f(r.get("nnva_span_sn"))),
                "Bos / bekleme (NVA)": r_bos}))
        else:
            gruplar.append((et, {
                "Islem suresi": sd(_f(r.get("va_span_sn"))
                                   + _f(r.get("nnva_span_sn"))),
                "Bos / bekleme": r_bos}))
    h.append('<div class="card"><div class="lgn">' + "".join(
        f'<span><i style="background:{c}"></i>{a}</span>' for a, c in seri) + "</div>"
        + _stacked(gruplar, seri, sb(), 0, etiket_her=max(1, len(son)//8), dar=True)
        + '<p class="alt" style="margin:10px 0 0">Her sutun bir parca — soldan '
          'saga zaman sirasi. Uzun sutunlar ve buyuyen bos bloklar tek tek '
          'incelenmeye deger.</p></div>')

    # ── adim sureleri ──
    h.append("<h2>Adim bazli ortalama sure (eleman suresi)</h2>")
    adim = sorted(((pretty(s), sd(v)) for s, v in o["ort_adim"].items()),
                  key=lambda x: -x[1])
    h.append('<div class="card">' + _hbars(adim, sb(), ondalik=1) + "</div>")

    # ── gunluk ozet ──
    tum = logger.gunler()[-gun_sayisi:]
    gunluk = []
    for g in tum:
        gr, _ = gecerli_satirlar(logger.load(g))
        z = _ozetle(gr)
        if z["adet"] > 0:
            gunluk.append((g, z))
    if gunluk:
        h.append(f"<h2>Gunluk ozet — son {len(gunluk)} gun</h2>")
        h.append('<div class="card">')
        h.append('<div class="lgn"><span><i style="background:var(--s1)"></i>'
                 f'Ortalama cevrim suresi ({sb()}/parca)</span></div>')
        h.append(_stacked([(g[5:], {sb(): sd(z["ort_cevrim"])}) for g, z in gunluk],
                          [(sb(), "var(--s1)")], sb(), 0))
        h.append("<table style='margin-top:18px'><thead><tr><th>Gun</th><th>Adet</th>"
                 f"<th>Ort. cevrim<br>({sb()})</th><th>parca/saat</th>"
                 + ("<th>VA %</th>" if va else "")
                 + "</tr></thead><tbody>")
        for g, z in gunluk:
            h.append(f"<tr><td>{g}</td><td>{z['adet']}</td><td>{sg(z['ort_cevrim'])}</td>"
                     f"<td>{nf(z['parca_saat'],1)}</td>"
                     + (f"<td>{nf(z['va_yuzde'],0)}</td>" if va else "")
                     + "</tr>")
        h.append("</tbody></table></div>")

    # ── eksik klipleri ──
    klipler = gun_klipleri(gun) if KLIP_AKTIF else []
    if klipler:
        h.append(f"<h2>Eksik islem klipleri — {len(klipler)} kayit</h2>")
        h.append('<div class="card"><table><thead><tr><th>Saat</th><th>Eksik adim</th>'
                 '<th>Video</th></tr></thead><tbody>')
        for f in klipler[:40]:
            parca = f[:-4].split("_")
            saat = parca[2].replace("-", ":") if len(parca) > 2 else ""
            eksik = " ".join(parca[3:]).replace("_", " ") if len(parca) > 3 else "-"
            tur = "ELLE" if f.startswith("elle") else eksik
            h.append(f'<tr><td>{saat}</td><td>{tur}</td>'
                     f'<td><a href="klipler/{f}" target="_blank">{f}</a></td></tr>')
        h.append("</tbody></table>"
                 '<p class="alt" style="margin:10px 0 0">Videolar kayitlar/klipler '
                 "klasorunde. Her klip TUM cevrimi degil, eksik adimin yapilmasi "
                 "gereken araligi kapsar: bir onceki islemin bitisi ile bir sonraki "
                 "islemin baslangici arasi. Yanindaki .txt dosyasinda hangi araligin "
                 "neden secildigi yazar.</p></div>")

    # ── yapay zeka analisti ──
    if AI_PANEL:
        try:
            veri = _rapor_verisi(gun, o, saat_ozet, vard, durus, rows, gunluk, va)
            h.append(_ai_panel(veri, _gemini_key()))
        except Exception as e:
            print("[rapor] AI paneli eklenemedi:", e)

    h.append('<p class="foot">Cevrim = parca postadan ciktigi andan bir sonraki '
             'parca ciktigi ana kadar; motor gelirken yapilan islemler de sayilir. '
             + ('Katma deger (VA) = motora parca takan islemler: '
                + ", ".join(pretty(x) for x in sorted(VA_OPERATIONS))
                + '. NVA = diger islemler ('
                + ", ".join(pretty(x) for x in NNVA_OPERATIONS)
                + ') + hic islem gorulmeyen bos sure. ' if va else
                'Islem suresi = izlenen islem etiketlerinin goruldugu sure ('
                + ", ".join(pretty(x) for x in IS_CLASSES)
                + '); geri kalan sure bos/bekleme. ')
             + 'Sureler bir islemin ilk goruldugu andan son goruldugu ana kadar '
               '(iki etiket arasi) olculur. '
             + f'{nf(CEVRIM_MAX_SN,0)} sn ustu cevrimler durus sayilir ve ortalamalara '
             'girmez. cmin = santidakika (1 dk = 100 cmin).</p>')
    return _sayfa(gun, "".join(h))


def _sayfa(gun, govde):
    # genis tablolar kartin disina tasmasin: kendi icinde yatay kaysin
    govde = govde.replace("<table", '<div class="tblwrap"><table')
    govde = govde.replace("</table>", "</table></div>")
    return ("<!doctype html><html lang='tr'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{URUN_ADI} — Posta uretim raporu {gun}</title>"
            f"<style>{_RAPOR_CSS}{_AI_CSS}</style>"
            f"</head><body><div class='wrap'>{govde}</div></body></html>")


_SRV = None


def rapor_sunucu(dizin, port=8777):
    """Raporu http://127.0.0.1:PORT uzerinden sunar.

    Tarayici, dosyadan (file://) acilan sayfalarda AI isteklerini engelleyebilir;
    yerel sunucu bu sorunu ortadan kaldirir. Sadece bu bilgisayardan erisilir.
    """
    global _SRV
    if _SRV is not None:
        return _SRV[1]
    from functools import partial
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
    for p in range(port, port + 20):
        try:
            h = partial(SimpleHTTPRequestHandler, directory=dizin)
            srv = ThreadingHTTPServer(("127.0.0.1", p), h)
        except OSError:
            continue
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        _SRV = (srv, p)
        print(f"[rapor] yerel sunucu: http://127.0.0.1:{p}/")
        return p
    raise OSError("uygun port bulunamadi")


def rapor_olustur(logger=None, gun=None, ac=True, sunucu=False, va=None):
    logger = logger or CycleLogger()
    gun = gun or datetime.now().strftime("%Y-%m-%d")
    html = rapor_html(logger, gun, va=va)
    p = os.path.join(logger.dir, f"rapor_{gun}.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[rapor] olusturuldu: {p}")
    if ac:
        try:
            if sunucu:
                port = rapor_sunucu(logger.dir)
                webbrowser.open(f"http://127.0.0.1:{port}/{os.path.basename(p)}")
            else:
                webbrowser.open("file:///" + p.replace("\\", "/"))
        except Exception as e:
            print("[rapor] acilamadi:", e)
    return p


@dataclass
class VideoState:
    fps: float = 30.0
    start_wall: Optional[float] = None
    start_frame: int = 0


class App:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{URUN_ADI} {SURUM} — {URUN_ALT}")
        self.root.configure(bg=TEMA["zemin"])
        self.root.geometry("460x120"); self.root.resizable(False, False)
        print(f"\n{'='*62}\n  {URUN_ADI} {SURUM} — {URUN_ALT}\n{'='*62}")
        loading = tk.Label(root, text=f"{URUN_ADI} — modeller yukleniyor...",
                           bg=TEMA["zemin"], fg=TEMA["cyan"], font=("Segoe UI", 11))
        loading.pack(expand=True); root.update()

        torch.set_num_threads(6)
        self.model_qa,     self.ov_qa     = self._load_yolo(QA_MODEL_PATH, "detect", required=True)
        self.model_pose,   self.ov_pose   = self._load_pose()
        self.model_sil,    self.ov_sil    = self._load_yolo(SILIKON_MODEL_PATH, "detect", required=False)
        self.model_musur,  self.ov_musur  = self._load_yolo(MUSUR_MODEL_PATH, "detect", required=False)
        if self.model_musur is not None:
            print(f"[musur] musur modeli yuklendi ({'OpenVINO' if self.ov_musur else 'PyTorch'}) "
                  f"siniflar: {list(self.model_musur.names.values())}")
        self.model_seg,    self.ov_seg    = (self._load_yolo(SEG_MODEL_PATH, "segment", required=False)
                                             if KISI_BULANIK else (None, False))
        print(f"[KVKK] maske: {'vucut silueti (segmentasyon)' if self.model_seg is not None else 'kisi kutusu'}"
              if KISI_BULANIK else "[KVKK] maske: yalnizca yuz")
        if self.model_sil is not None:
            print(f"[silikon] spatula modeli yuklendi ({'OpenVINO' if self.ov_sil else 'PyTorch'}) "
                  f"siniflar: {list(self.model_sil.names.values())}")

        # TESHIS: model siniflari QA adimlariyla esleşiyor mu
        self._report_classes()

        loading.destroy()
        self.root.geometry("1500x940"); self.root.resizable(True, True)
        self.root.minsize(1100, 700)

        self.conf_qa     = tk.DoubleVar(value=0.40)   # genel islem modeli
        # ayri conf'lar
        self.conf_kam    = tk.DoubleVar(value=PER_CLASS_CONF["kam_mili koyma"])
        self.conf_musur  = tk.DoubleVar(value=PER_CLASS_CONF["musur_skma"])
        self.conf_motor  = tk.DoubleVar(value=MOTOR_CONF)
        # ayri onay sureleri (dwell)
        self.dwell_kam   = tk.DoubleVar(value=STEP_DWELL["kam_mili koyma"])
        self.dwell_musur = tk.DoubleVar(value=STEP_DWELL["musur_skma"])
        # motor cikis esigi + okluzyon toleransi
        self.motor_move  = tk.DoubleVar(value=MOTOR_MOVE_THRES)
        self.motor_hold  = tk.DoubleVar(value=MOTOR_HOLD_SECS)
        self.motor_lost  = tk.DoubleVar(value=MOTOR_LOST_SECS)
        self.motor_absent_closes = tk.BooleanVar(value=MOTOR_ABSENT_CLOSES)

        self.imgsz_idx = tk.IntVar(value=IMGSZ_CHOICES.index(640))
        self.alpha = tk.DoubleVar(value=0.50)
        self.frame_skip = tk.IntVar(value=2)
        self.paused = tk.BooleanVar(value=False)
        self.show_pose = tk.BooleanVar(value=True)
        self.yuz_bulanik = tk.BooleanVar(value=YUZ_BULANIK)
        self.tk_video = tk.BooleanVar(value=True)      # Tk'de goruntu goster
        self._tk_son_kare = 0.0                        # son cizim zamani
        self._web_son_durum = 0.0
        self._web_ayar_kes = ({}, 0.0)
        self.yuz_kabalik = tk.IntVar(value=YUZ_MOZAIK)
        self._yuz_son = None
        # raporda katma deger (VA/NVA) ayrimi gosterilsin mi (opsiyonel)
        self.rapor_va = tk.BooleanVar(value=VA_RAPOR)
        self.show_zones = tk.BooleanVar(value=True)
        self.show_motor_gate = tk.BooleanVar(value=True)
        self.edit_roi = tk.BooleanVar(value=False)
        self.source_label = tk.StringVar(value="(none)")
        self.fps_text = tk.StringVar(value="0.0 fps")
        self.det_text = tk.StringVar(value="Det: 0")
        self.motor_text = tk.StringVar(value="Motor: -")
        self.hands_text = tk.StringVar(value="Kisi: 0   El noktasi: 0")
        self.ip_url = tk.StringVar(value="rtsp://admin:Bakim789456_@192.168.1.13:554/media/video1")
        self.ip_status = tk.StringVar(value="")

        # ── uretim kaydi ──
        self.logger = CycleLogger()
        self.clips = ClipBuffer()
        self.klip_ac = tk.BooleanVar(value=KLIP_KAYDET)
        self.klip_text = tk.StringVar(value="Klip tamponu: —")
        self.cycle_rows = self.logger.load(datetime.now().strftime("%Y-%m-%d"))
        self.rapor_text = tk.StringVar(value=f"kayitlar/ — {len(self.cycle_rows)} cevrim")

        self.zones = [ZoneMonitor(z) for z in ROI_ZONES]
        self.zone_lock = threading.Lock()
        self.qa_reset_evt = threading.Event()
        self.qa_labels = {}
        self.zone_labels = {}
        self.disp_geom = None      # (off_x, off_y, scale, frame_w, frame_h)
        self.drag_start = None
        self.drag_rect = None

        self._build_ui()
        self.web_q = queue.Queue()
        self.web = WebPanel(self) if WEB_PANEL else None
        if self.web:
            try:
                self.web.basla()
                if hasattr(self, "web_adres"):
                    self.web_adres.config(text=self.web.adres)
            except Exception as e:
                print("[web] panel baslatilamadi:", e); self.web = None
        self.last_annotated = None
        self.cmd_q = queue.Queue()
        self.alive = True
        self.worker = threading.Thread(target=self._worker_main, daemon=True)
        self.worker.start()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(200, self._web_kuyruk_isle)
        self.root.after(300, self.connect_ip)

    # ── model yukleme ───────────────────────────────────────────────────
    def _load_yolo(self, path, task, required):
        if not os.path.exists(path):
            if required: raise FileNotFoundError(f"Model bulunamadi: {path}")
            print(f"[uyari] Model yok, ozellik kapali: {path}")
            return None, False
        ov_dir = path.replace(".pt", "_openvino_model")
        if DEVICE == "cpu":
            if not os.path.isdir(ov_dir):
                try: YOLO(path).export(format="openvino", dynamic=True)
                except Exception: pass
            if os.path.isdir(ov_dir):
                try:
                    m = YOLO(ov_dir, task=task)
                    m.predict(np.zeros((320,320,3), np.uint8), imgsz=640, verbose=False)
                    return m, True
                except Exception: pass
        m = YOLO(path)
        try: m.to(DEVICE)
        except Exception: pass
        try: m.fuse()
        except Exception: pass
        m.predict(np.zeros((320,320,3), np.uint8), imgsz=640, device=DEVICE, half=HALF, verbose=False)
        return m, False

    def _report_classes(self):
        """Modelin siniflarini ve QA adimlariyla eslesmeyi konsola yazar."""
        print("\n" + "="*60)
        if self.model_qa is None:
            print(f"[!] QA MODELI YUKLENMEDI: {QA_MODEL_PATH}")
            print("    Yeni egittigin best.pt dosyasini bu isimle bu klasore koy.")
            print("="*60 + "\n")
            return
        names = list(self.model_qa.names.values())
        print(f"[model] QA modeli siniflari ({len(names)}): {names}")
        eksik = [s for s in QA_STEP_CLASSES if s not in names and s not in SANAL_ADIMLAR]
        if eksik:
            print("\n[!] DIKKAT — bu QA adimlari modelde YOK, asla isaretlenmez:")
            for s in eksik:
                print(f"    - '{s}'")
            print("    QA_STEP_CLASSES listesindeki yazimi yukaridaki liste ile birebir esitle.")
        else:
            print("\n[ok] Tum QA adimlari modelde mevcut.")
        if MOTOR_CLASS not in names:
            print(f"\n[!] KRITIK — '{MOTOR_CLASS}' sinifi modelde YOK.")
            print("    Cevrim kapatma motor hareketine bagli; motor tespit edilemezse")
            print("    QA hicbir zaman otomatik kapanmaz/sifirlanmaz.")
        else:
            print(f"[ok] Cevrim kapatma sinifi '{MOTOR_CLASS}' modelde mevcut.")
        print("="*60 + "\n")

    def _load_pose(self):
        # v8: iskelet modeli de OpenVINO ile calisir (Intel CPU'da PyTorch'a gore
        # ~3 kat hizli: olcumde 72 ms -> 23 ms/kare). Ilk acilista otomatik
        # donusturulur; donusum basarisiz olursa eski PyTorch yoluna duser.
        yol = POSE_MODEL_PATH if os.path.isabs(POSE_MODEL_PATH) else os.path.join(BASE_DIR, POSE_MODEL_PATH)
        if os.path.exists(yol):
            try:
                m, ov = self._load_yolo(yol, "pose", required=False)
                if m is not None:
                    print(f"[pose] iskelet modeli yuklendi ({'OpenVINO' if ov else 'PyTorch'})")
                    return m, ov
            except Exception as e:
                print(f"[uyari] Pose OpenVINO yuklenemedi, PyTorch'a donuluyor: {e}")
        try:
            m = YOLO(POSE_MODEL_PATH)
            try: m.to(DEVICE)
            except Exception: pass
            m.predict(np.zeros((320,320,3), np.uint8), imgsz=640, device=DEVICE, half=HALF, verbose=False)
            return m, False
        except Exception as e:
            print(f"[uyari] Pose modeli yuklenemedi: {e}")
            return None, False

    def _pk(self, using_ov, imgsz, conf):
        k = dict(imgsz=imgsz, conf=conf, verbose=False)
        if not using_ov: k.update(device=DEVICE, half=HALF)
        return k

    def _live_conf_map(self):
        """Sinif bazinda canli confidence esikleri."""
        m = dict(PER_CLASS_CONF)
        m["kam_mili koyma"] = float(self.conf_kam.get())
        m["musur_skma"]     = float(self.conf_musur.get())
        m[MOTOR_CLASS]      = float(self.conf_motor.get())
        return m

    def _live_dwell_map(self):
        """Adim bazinda canli onay sureleri."""
        m = dict(STEP_DWELL)
        m["kam_mili koyma"] = float(self.dwell_kam.get())
        m["musur_skma"]     = float(self.dwell_musur.get())
        return m

    # ══ ARAYUZ ═══════════════════════════════════════════════════════════
    def _build_ui(self):
        self._apply_theme()
        self.root.configure(bg=TEMA["zemin"])

        kok = tk.Frame(self.root, bg=TEMA["zemin"])
        kok.pack(fill="both", expand=True)

        # ── sol panel (sabit genislik, kaydirilabilir) ──
        sol = tk.Frame(kok, bg=TEMA["zemin"], width=PANEL_W)
        sol.pack(side="left", fill="y"); sol.pack_propagate(False)

        self._build_baslik(sol)

        govde = tk.Frame(sol, bg=TEMA["zemin"])
        govde.pack(fill="both", expand=True, padx=10)
        self.sb_canvas = tk.Canvas(govde, bg=TEMA["zemin"], highlightthickness=0,
                                   borderwidth=0)
        vs = ttk.Scrollbar(govde, orient="vertical", command=self.sb_canvas.yview,
                           style="Ince.Vertical.TScrollbar")
        self.sb_canvas.configure(yscrollcommand=vs.set)
        vs.pack(side="right", fill="y"); self.sb_canvas.pack(side="left", fill="both", expand=True)
        ic = tk.Frame(self.sb_canvas, bg=TEMA["zemin"])
        self._sb_window = self.sb_canvas.create_window((0, 0), window=ic, anchor="nw")
        ic.bind("<Configure>",
                lambda e: self.sb_canvas.configure(scrollregion=self.sb_canvas.bbox("all")))
        self.sb_canvas.bind("<Configure>",
                            lambda e: self.sb_canvas.itemconfig(self._sb_window, width=e.width))
        def _wheel(e): self.sb_canvas.yview_scroll(int(-e.delta / 120), "units")
        self.sb_canvas.bind("<Enter>", lambda e: self.sb_canvas.bind_all("<MouseWheel>", _wheel))
        self.sb_canvas.bind("<Leave>", lambda e: self.sb_canvas.unbind_all("<MouseWheel>"))

        # ── sekmeler ──
        self.nb = ttk.Notebook(ic, style="Ana.TNotebook")
        self.nb.pack(fill="both", expand=True)
        s_canli   = tk.Frame(self.nb, bg=TEMA["zemin"])
        s_uretim  = tk.Frame(self.nb, bg=TEMA["zemin"])
        s_ayar    = tk.Frame(self.nb, bg=TEMA["zemin"])
        self.nb.add(s_canli,  text="  CANLI  ")
        self.nb.add(s_uretim, text="  URETIM  ")
        self.nb.add(s_ayar,   text="  AYARLAR  ")

        self._sekme_canli(s_canli)
        self._sekme_uretim(s_uretim)
        self._sekme_ayarlar(s_ayar)

        self._build_durum_serit(sol)
        self._build_canvas(kok)

    # ── tema ────────────────────────────────────────────────────────────
    def _apply_theme(self):
        st = ttk.Style()
        try: st.theme_use("clam")
        except Exception: pass
        T = TEMA
        st.configure(".", background=T["zemin"], foreground=T["metin"],
                     fieldbackground=T["panel"], bordercolor=T["cizgi"],
                     lightcolor=T["panel"], darkcolor=T["panel"])
        st.configure("TFrame", background=T["zemin"])
        st.configure("TLabel", background=T["panel"], foreground=T["metin"])
        st.configure("TCheckbutton", background=T["panel"], foreground=T["soluk"],
                     focuscolor=T["panel"], indicatorbackground=T["panel2"],
                     indicatorforeground=T["cyan"], bordercolor=T["cizgi"],
                     indicatormargin=(0, 0, 6, 0), padding=(0, 3))
        st.map("TCheckbutton", background=[("active", T["panel"])],
               foreground=[("active", T["metin"])])
        st.configure("TEntry", fieldbackground=T["panel2"], foreground=T["metin"],
                     insertcolor=T["cyan"], bordercolor=T["cizgi"])
        # sekmeler
        st.configure("Ana.TNotebook", background=T["zemin"], borderwidth=0, tabmargins=0)
        st.configure("Ana.TNotebook.Tab", background=T["zemin"], foreground=T["cok_soluk"],
                     padding=(10, 7), borderwidth=0, font=("Segoe UI", 9, "bold"))
        st.map("Ana.TNotebook.Tab",
               background=[("selected", T["panel"])],
               foreground=[("selected", T["cyan"]), ("active", T["metin"])])
        # sliderlar
        st.configure("Ince.Horizontal.TScale", background=T["panel"],
                     troughcolor=T["panel2"], bordercolor=T["cizgi"],
                     lightcolor=T["cyan"], darkcolor=T["cyan"])
        st.configure("Ince.Vertical.TScrollbar", background=T["cizgi"],
                     troughcolor=T["zemin"], bordercolor=T["zemin"],
                     arrowcolor=T["cok_soluk"], width=8)

    # ── kucuk yapi taslari ──────────────────────────────────────────────
    def _kart(self, parent, baslik=None, ikon=""):
        """Basligi olan koyu kart."""
        dis = tk.Frame(parent, bg=TEMA["cizgi"])
        dis.pack(fill="x", pady=(0, 8))
        k = tk.Frame(dis, bg=TEMA["panel"])
        k.pack(fill="both", padx=1, pady=1)
        if baslik:
            ust = tk.Frame(k, bg=TEMA["panel"])
            ust.pack(fill="x", padx=10, pady=(8, 2))
            tk.Label(ust, text=f"{ikon} {baslik}".strip(), bg=TEMA["panel"],
                     fg=TEMA["cok_soluk"], font=("Segoe UI", 7, "bold")).pack(anchor="w")
        return k

    def _btn(self, parent, text, cmd, tur="normal"):
        renk = {"birincil": (TEMA["cyan"], "#06202b"),
                "ikincil":  (TEMA["panel2"], TEMA["metin"]),
                "normal":   (TEMA["panel2"], TEMA["soluk"])}[tur]
        b = tk.Button(parent, text=text, command=cmd, bg=renk[0], fg=renk[1],
                      activebackground=TEMA["mor"] if tur == "birincil" else TEMA["cizgi"],
                      activeforeground="#ffffff", relief="flat", bd=0,
                      font=("Segoe UI", 9, "bold" if tur == "birincil" else "normal"),
                      cursor="hand2", pady=6, highlightthickness=0)
        return b

    def _add_slider(self, parent, initial, var, from_, to, fmt):
        satir = tk.Frame(parent, bg=TEMA["panel"])
        satir.pack(fill="x", padx=10, pady=(6, 0))
        lbl = tk.Label(satir, text=initial, bg=TEMA["panel"], fg=TEMA["soluk"],
                       font=("Segoe UI", 8), anchor="w")
        lbl.pack(fill="x")
        ttk.Scale(satir, from_=from_, to=to, variable=var, style="Ince.Horizontal.TScale",
                  command=lambda v, l=lbl: l.config(text=fmt(v))).pack(fill="x", pady=(1, 3))

    # ── baslik seridi ───────────────────────────────────────────────────
    def _build_baslik(self, parent):
        b = tk.Frame(parent, bg=TEMA["panel"])
        b.pack(fill="x", padx=10, pady=(10, 8))
        ic = tk.Frame(b, bg=TEMA["panel"]); ic.pack(fill="x", padx=10, pady=9)
        try:
            if os.path.exists(AMBLEM_PATH) or os.path.exists(LOGO_PATH):
                im = Image.open(AMBLEM_PATH if os.path.exists(AMBLEM_PATH) else LOGO_PATH)
                im.thumbnail((38, 38), Image.LANCZOS)
                self._logo_img = ImageTk.PhotoImage(im)
                tk.Label(ic, image=self._logo_img, bg=TEMA["panel"]).pack(side="left",
                                                                         padx=(0, 9))
        except Exception as e:
            print("[marka] logo yuklenemedi:", e)
        yazi = tk.Frame(ic, bg=TEMA["panel"]); yazi.pack(side="left", fill="x", expand=True)
        tk.Label(yazi, text=URUN_ADI, bg=TEMA["panel"], fg=TEMA["metin"],
                 font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(yazi, text=URUN_ALT, bg=TEMA["panel"], fg=TEMA["cok_soluk"],
                 font=("Segoe UI", 6)).pack(anchor="w")
        sag = tk.Frame(ic, bg=TEMA["panel"]); sag.pack(side="right")
        self.canli_nokta = tk.Canvas(sag, width=10, height=10, bg=TEMA["panel"],
                                     highlightthickness=0)
        self.canli_nokta.pack(side="right")
        self._nokta = self.canli_nokta.create_oval(1, 1, 9, 9, fill=TEMA["cok_soluk"],
                                                   outline="")
        self.fps_lbl = tk.Label(sag, textvariable=self.fps_text, bg=TEMA["panel"],
                                fg=TEMA["cyan"], font=("Consolas", 8))
        self.fps_lbl.pack(side="right", padx=(0, 6))

    # ── alt durum seridi ────────────────────────────────────────────────
    def _build_durum_serit(self, parent):
        s = tk.Frame(parent, bg=TEMA["panel2"])
        s.pack(fill="x", side="bottom")
        tk.Label(s, textvariable=self.source_label, bg=TEMA["panel2"],
                 fg=TEMA["cok_soluk"], font=("Segoe UI", 7), anchor="w",
                 wraplength=PANEL_W-20).pack(fill="x", padx=10, pady=(5, 0))
        tk.Label(s, textvariable=self.det_text, bg=TEMA["panel2"], fg=TEMA["cok_soluk"],
                 font=("Segoe UI", 7), anchor="w").pack(fill="x", padx=10)
        tk.Label(s, textvariable=self.hands_text, bg=TEMA["panel2"], fg=TEMA["cok_soluk"],
                 font=("Segoe UI", 7), anchor="w").pack(fill="x", padx=10, pady=(0, 5))

    # ══ SEKME: CANLI ═════════════════════════════════════════════════════
    def _sekme_canli(self, p):
        # ── QA durumu: halka + adimlar ──
        k = self._kart(p, "ISLEM KONTROL LISTESI", "◉")
        ust = tk.Frame(k, bg=TEMA["panel"]); ust.pack(fill="x", padx=10, pady=(2, 4))
        self.halka = tk.Canvas(ust, width=86, height=86, bg=TEMA["panel"],
                               highlightthickness=0)
        self.halka.pack(side="left", padx=(0, 10))
        self._halka_ciz(0, 5, TEMA["cok_soluk"])
        sag = tk.Frame(ust, bg=TEMA["panel"]); sag.pack(side="left", fill="both", expand=True)
        self.qa_durum = tk.Label(sag, text="BEKLIYOR", bg=TEMA["panel"], fg=TEMA["sari"],
                                 font=("Segoe UI", 15, "bold"), anchor="w")
        self.qa_durum.pack(fill="x")
        self.qa_alt = tk.Label(sag, text="motor bekleniyor", bg=TEMA["panel"],
                               fg=TEMA["soluk"], font=("Segoe UI", 8), anchor="w",
                               justify="left", wraplength=PANEL_W-150)
        self.qa_alt.pack(fill="x")

        liste = tk.Frame(k, bg=TEMA["panel"]); liste.pack(fill="x", padx=10, pady=(2, 8))
        self.qa_labels = {}
        for step in QA_STEP_CLASSES:
            satir = tk.Frame(liste, bg=TEMA["panel"]); satir.pack(fill="x", pady=1)
            nk = tk.Canvas(satir, width=8, height=8, bg=TEMA["panel"], highlightthickness=0)
            nk.pack(side="left", padx=(0, 7)); nk.create_oval(1, 1, 7, 7,
                                                             fill=TEMA["cizgi"], outline="")
            lbl = tk.Label(satir, text=pretty(step), bg=TEMA["panel"], fg=TEMA["cok_soluk"],
                           font=("Segoe UI", 9), anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            ek = tk.Label(satir, text="", bg=TEMA["panel"], fg=TEMA["cok_soluk"],
                          font=("Segoe UI", 7))
            ek.pack(side="right")
            self.qa_labels[step] = (nk, lbl, ek)
        tk.Label(liste, text="* motor gelirken de yapilabilir", bg=TEMA["panel"],
                 fg=TEMA["cok_soluk"], font=("Segoe UI", 7)).pack(anchor="w", pady=(4, 0))

        # ── motor durumu ──
        k = self._kart(p, "MOTOR CEVRIMI", "⬢")
        m = tk.Frame(k, bg=TEMA["panel"]); m.pack(fill="x", padx=10, pady=(2, 9))
        self.motor_nokta = tk.Canvas(m, width=10, height=10, bg=TEMA["panel"],
                                     highlightthickness=0)
        self.motor_nokta.pack(side="left", padx=(0, 8))
        self._mnokta = self.motor_nokta.create_oval(0, 0, 10, 10, fill=TEMA["cok_soluk"],
                                                    outline="")
        msag = tk.Frame(m, bg=TEMA["panel"]); msag.pack(side="left", fill="x", expand=True)
        self.motor_ad = tk.Label(msag, text="MOTOR YOK", bg=TEMA["panel"], fg=TEMA["soluk"],
                                 font=("Segoe UI", 11, "bold"), anchor="w")
        self.motor_ad.pack(fill="x")
        self.motor_alt = tk.Label(msag, textvariable=self.motor_text, bg=TEMA["panel"],
                                  fg=TEMA["cok_soluk"], font=("Consolas", 8), anchor="w")
        self.motor_alt.pack(fill="x")
        self.motor_bar = tk.Canvas(k, height=4, bg=TEMA["panel2"], highlightthickness=0)
        self.motor_bar.pack(fill="x", padx=10, pady=(0, 10))

        # ── bolgeler ──
        k = self._kart(p, "BOLGELER (ROI)", "▣")
        self.zone_labels = {}
        for z in self.zones:
            satir = tk.Frame(k, bg=TEMA["panel"]); satir.pack(fill="x", padx=10, pady=1)
            nk = tk.Canvas(satir, width=8, height=8, bg=TEMA["panel"], highlightthickness=0)
            nk.pack(side="left", padx=(0, 7)); nk.create_oval(1, 1, 7, 7,
                                                             fill=TEMA["cizgi"], outline="")
            lbl = tk.Label(satir, text=z.name, bg=TEMA["panel"], fg=TEMA["cok_soluk"],
                           font=("Segoe UI", 9), anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            ek = tk.Label(satir, text="", bg=TEMA["panel"], fg=TEMA["cok_soluk"],
                          font=("Segoe UI", 7)); ek.pack(side="right")
            self.zone_labels[z.name] = (nk, lbl, ek)
        ttk.Checkbutton(k, text="ROI duzenle (goruntude surukle)",
                        variable=self.edit_roi).pack(anchor="w", padx=10, pady=(6, 2))
        self._btn(k, "Bolge tetiklerini sifirla", self.reset_zones).pack(fill="x", padx=10,
                                                                        pady=(0, 10))

        # ── hizli kontroller ──
        k = self._kart(p, "KONTROL", "⏻")
        g = tk.Frame(k, bg=TEMA["panel"]); g.pack(fill="x", padx=10, pady=(2, 10))
        self._btn(g, "⏸  Duraklat / Devam", self.toggle_pause).pack(fill="x", pady=2)
        self._btn(g, "⭯  QA sifirla (yeni parca)", self.reset_qa, "ikincil").pack(fill="x",
                                                                                  pady=2)
        self._btn(g, "⤓  Anlik goruntu", self.snapshot).pack(fill="x", pady=2)
        self._btn(g, "⏹  Durdur", lambda: self.cmd_q.put(("stop",))).pack(fill="x", pady=2)

    def _halka_ciz(self, oran, toplam, renk):
        """QA ilerlemesini halka olarak cizer."""
        c = self.halka; c.delete("all")
        c.create_oval(6, 6, 80, 80, outline=TEMA["panel2"], width=8)
        if oran > 0:
            c.create_arc(6, 6, 80, 80, start=90, extent=-359.9 * oran, style="arc",
                         outline=renk, width=8)
        c.create_text(43, 39, text=f"{int(round(oran * toplam))}/{toplam}",
                      fill=TEMA["metin"], font=("Segoe UI", 13, "bold"))
        c.create_text(43, 56, text="ADIM", fill=TEMA["cok_soluk"],
                      font=("Segoe UI", 6, "bold"))

    def _bar_ciz(self, canvas, oran, renk):
        canvas.delete("all")
        w = canvas.winfo_width()
        if w <= 2:                       # yerlesim henuz olusmadiysa
            w = max(canvas.winfo_reqwidth(), PANEL_W - 60)
        h = int(canvas["height"])
        canvas.create_rectangle(0, 0, int(w * max(0.0, min(1.0, oran))), h,
                                fill=renk, width=0)

    # ══ SEKME: URETIM ════════════════════════════════════════════════════
    def _sekme_uretim(self, p):
        k = self._kart(p, "AKTIF CEVRIM", "⏱")
        a = tk.Frame(k, bg=TEMA["panel"]); a.pack(fill="x", padx=10, pady=(2, 10))
        self.aktif_buyuk = tk.Label(a, text="—", bg=TEMA["panel"], fg=TEMA["cyan"],
                                    font=("Segoe UI", 26, "bold"), anchor="w")
        self.aktif_buyuk.pack(fill="x")
        self.aktif_alt = tk.Label(a, text="motor bekleniyor", bg=TEMA["panel"],
                                  fg=TEMA["cok_soluk"], font=("Segoe UI", 8), anchor="w")
        self.aktif_alt.pack(fill="x")

        # KPI kartlari (2 sutun)
        k = self._kart(p, "GUN OZETI", "▤")
        g = tk.Frame(k, bg=TEMA["panel"]); g.pack(fill="x", padx=8, pady=(2, 10))
        g.columnconfigure((0, 1), weight=1, uniform="kpi")
        self.kpi = {}
        for i, (anahtar, ad) in enumerate((("son", "SON CEVRIM"), ("ort", "GUN ORT."),
                                           ("hiz", "HIZ"), ("adet", "PARCA"))):
            hucre = tk.Frame(g, bg=TEMA["panel2"])
            hucre.grid(row=i // 2, column=i % 2, sticky="nsew", padx=2, pady=2)
            tk.Label(hucre, text=ad, bg=TEMA["panel2"], fg=TEMA["cok_soluk"],
                     font=("Segoe UI", 6, "bold"), anchor="w").pack(fill="x", padx=8,
                                                                    pady=(6, 0))
            v = tk.Label(hucre, text="—", bg=TEMA["panel2"], fg=TEMA["metin"],
                         font=("Segoe UI", 14, "bold"), anchor="w")
            v.pack(fill="x", padx=8)
            alt = tk.Label(hucre, text="", bg=TEMA["panel2"], fg=TEMA["cok_soluk"],
                           font=("Segoe UI", 7), anchor="w")
            alt.pack(fill="x", padx=8, pady=(0, 6))
            self.kpi[anahtar] = (v, alt)

        k = self._kart(p, "RAPOR", "◱")
        r = tk.Frame(k, bg=TEMA["panel"]); r.pack(fill="x", padx=10, pady=(2, 10))
        ttk.Checkbutton(r, text="Raporda katma deger (VA/NVA) ayrimi",
                        variable=self.rapor_va).pack(anchor="w", pady=(0, 4))
        self._btn(r, "◱  Rapor olustur (HTML)", self.rapor, "birincil").pack(fill="x", pady=2)
        self._btn(r, "✦  Rapor + AI analisti", lambda: self.rapor(sunucu=True),
                  "ikincil").pack(fill="x", pady=2)
        self._btn(r, "🗀  Kayit klasoru", self.kayit_klasor).pack(fill="x", pady=2)
        if WEB_PANEL:
            self._btn(r, "🖵  Web panelini ac", self.web_ac, "ikincil").pack(fill="x", pady=2)
            self._btn(r, "✦  Tanitim sayfasi", self.tanitim_ac).pack(fill="x", pady=2)
            self.web_adres = tk.Label(r, text="", bg=TEMA["panel"], fg=TEMA["cyan"],
                                      font=("Consolas", 8), anchor="w")
            self.web_adres.pack(fill="x", pady=(2, 0))
        if KLIP_AKTIF:
            ttk.Checkbutton(r, text="Eksik cikinca klip kaydet",
                            variable=self.klip_ac).pack(anchor="w", pady=(6, 0))
            tk.Label(r, textvariable=self.klip_text, bg=TEMA["panel"], fg=TEMA["cok_soluk"],
                     font=("Segoe UI", 7), anchor="w").pack(fill="x")
            self._btn(r, f"Son {KLIP_ELLE_SN:.0f} sn'yi kaydet",
                      self.klip_simdi).pack(fill="x", pady=2)
        tk.Label(r, textvariable=self.rapor_text, bg=TEMA["panel"], fg=TEMA["cok_soluk"],
                 font=("Segoe UI", 7), anchor="w", wraplength=PANEL_W-40,
                 justify="left").pack(fill="x", pady=(4, 0))

    # ══ SEKME: AYARLAR ═══════════════════════════════════════════════════
    def _sekme_ayarlar(self, p):
        # kaynak
        k = self._kart(p, "KAYNAK", "⛁")
        g = tk.Frame(k, bg=TEMA["panel"]); g.pack(fill="x", padx=10, pady=(2, 8))
        tk.Label(g, text="IP kamera (RTSP)", bg=TEMA["panel"], fg=TEMA["cok_soluk"],
                 font=("Segoe UI", 7)).pack(anchor="w")
        ttk.Entry(g, textvariable=self.ip_url).pack(fill="x", pady=(2, 4))
        self._btn(g, "⇄  Baglan", self.connect_ip, "birincil").pack(fill="x")
        tk.Label(g, textvariable=self.ip_status, bg=TEMA["panel"], fg=TEMA["cok_soluk"],
                 font=("Segoe UI", 7), anchor="w").pack(fill="x", pady=(3, 6))
        sat = tk.Frame(g, bg=TEMA["panel"]); sat.pack(fill="x")
        for i in range(2):
            self._btn(sat, f"Webcam {i}",
                      lambda n=i: self.request_source(("cam", n), f"Webcam {n}")
                      ).pack(side="left", fill="x", expand=True, padx=(0 if i == 0 else 3, 0))
        self._btn(g, "▸  Video sec...", self.browse).pack(fill="x", pady=(3, 0))
        if os.path.isdir(VIDEO_DIR):
            for ad in sorted(f for f in os.listdir(VIDEO_DIR)
                             if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")))[:6]:
                yol = os.path.join(VIDEO_DIR, ad)
                self._btn(g, ad, lambda p_=yol, n=ad: self.request_source(("video", p_), n)
                          ).pack(fill="x", pady=1)

        # gorunum
        k = self._kart(p, "GORUNUM", "◈")
        g = tk.Frame(k, bg=TEMA["panel"]); g.pack(fill="x", padx=10, pady=(2, 8))
        for metin, degisken in (("Bu pencerede goruntu (kapatmak hizlandirir)",
                                 self.tk_video),
                                ("Iskelet (pose)", self.show_pose),
                                ("Kisileri gizle (KVKK)", self.yuz_bulanik),
                                ("Bolgeleri goster", self.show_zones),
                                ("Motor cevrim gostergesi", self.show_motor_gate)):
            ttk.Checkbutton(g, text=metin, variable=degisken).pack(anchor="w")
        self._add_slider(k, f"Mozaik kabaligi: {self.yuz_kabalik.get()}", self.yuz_kabalik,
                         6, 30, lambda v: f"Mozaik kabaligi: {int(float(v))}")
        self._add_slider(k, f"Goruntu boyutu: {IMGSZ_CHOICES[self.imgsz_idx.get()]}",
                         self.imgsz_idx, 0, len(IMGSZ_CHOICES) - 1,
                         lambda v: f"Goruntu boyutu: {IMGSZ_CHOICES[int(float(v))]}")
        self._add_slider(k, f"Kare atlama: {self.frame_skip.get()}", self.frame_skip, 0, 5,
                         lambda v: f"Kare atlama: {int(float(v))}")
        self._add_slider(k, f"Maske saydamligi: {self.alpha.get():.2f}", self.alpha, 0.0, 1.0,
                         lambda v: f"Maske saydamligi: {float(v):.2f}")
        tk.Frame(k, bg=TEMA["panel"], height=6).pack()

        # esikler
        k = self._kart(p, "TESPIT ESIKLERI", "◎")
        self._add_slider(k, f"Genel islem conf: {self.conf_qa.get():.2f}", self.conf_qa,
                         0.01, 0.99, lambda v: f"Genel islem conf: {float(v):.2f}")
        self._add_slider(k, f"Kam mili conf: {self.conf_kam.get():.2f}", self.conf_kam,
                         0.01, 0.99, lambda v: f"Kam mili conf: {float(v):.2f}")
        self._add_slider(k, f"Kam mili onay: {self.dwell_kam.get():.1f} sn", self.dwell_kam,
                         0.0, 3.0, lambda v: f"Kam mili onay: {float(v):.1f} sn")
        self._add_slider(k, f"Musur sikma conf: {self.conf_musur.get():.2f}", self.conf_musur,
                         0.01, 0.99, lambda v: f"Musur sikma conf: {float(v):.2f}")
        self._add_slider(k, f"Musur sikma onay: {self.dwell_musur.get():.1f} sn",
                         self.dwell_musur, 0.0, 3.0,
                         lambda v: f"Musur sikma onay: {float(v):.1f} sn")
        tk.Frame(k, bg=TEMA["panel"], height=6).pack()

        # motor
        k = self._kart(p, "MOTOR CEVRIMI", "⬢")
        self._add_slider(k, f"Motor conf: {self.conf_motor.get():.2f}", self.conf_motor,
                         0.01, 0.99, lambda v: f"Motor conf: {float(v):.2f}")
        self._add_slider(k, f"Cikis kayma esigi: {self.motor_move.get():.2f}",
                         self.motor_move, 0.02, 0.50,
                         lambda v: f"Cikis kayma esigi: {float(v):.2f}")
        self._add_slider(k, f"Okluzyon toleransi: {self.motor_hold.get():.1f} sn",
                         self.motor_hold, 0.5, 20.0,
                         lambda v: f"Okluzyon toleransi: {float(v):.1f} sn")
        ttk.Checkbutton(k, text="Uzun sure yoksa da kapat",
                        variable=self.motor_absent_closes).pack(anchor="w", padx=10)
        self._add_slider(k, f"Yokluk kapatma: {self.motor_lost.get():.0f} sn",
                         self.motor_lost, 2.0, 60.0,
                         lambda v: f"Yokluk kapatma: {float(v):.0f} sn")
        tk.Label(k, text="Motor yerlesince cevrim baslar; sadece gorulerek\ncapadan kayinca "
                         "kapanir. Etiket kaybolursa bozulmaz.",
                 bg=TEMA["panel"], fg=TEMA["cok_soluk"], font=("Segoe UI", 7),
                 justify="left", anchor="w").pack(fill="x", padx=10, pady=(4, 10))

    def _build_canvas(self, parent):
        cerceve = tk.Frame(parent, bg=TEMA["zemin"])
        cerceve.pack(side="right", fill="both", expand=True, padx=(0, 10), pady=10)
        self.canvas = tk.Label(cerceve, bg="#05080c", bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    # ══ WEB PANEL KOPRUSU ════════════════════════════════════════════════
    def _web_durum(self, st, m_state, qa_durum, qa_alt, _renk):
        """Tarayici paneline gonderilen anlik durum."""
        adimlar = []
        qa_prog = st.get("qa_prog", {}); qa_wait = st.get("qa_wait", {})
        for adim in QA_STEP_CLASSES:
            if st["qa_done"].get(adim):
                src = st["qa_src"].get(adim, "")
                adimlar.append({"ad": pretty(adim), "durum": "tamam",
                                "rozet": {"bolge": "bolge",
                                          "on": "motor oncesi"}.get(src, "model")})
            elif adim in qa_wait:
                adimlar.append({"ad": pretty(adim), "durum": "kilitli",
                                "rozet": qa_wait[adim]})
            elif qa_prog.get(adim, 0) > 0:
                adimlar.append({"ad": pretty(adim), "durum": "ilerliyor",
                                "rozet": f"%{qa_prog[adim]*100:.0f}",
                                "oran": qa_prog[adim]})
            else:
                adimlar.append({"ad": pretty(adim) + (" *" if adim in PRE_CYCLE_STEPS else ""),
                                "durum": "bekliyor", "rozet": ""})
        bolgeler = []
        for zad, (ic, trig, prog) in st.get("zones", {}).items():
            bolgeler.append({"ad": zad,
                             "durum": "tamam" if trig else ("ilerliyor" if ic else "bos"),
                             "etiket": "KAPI ACIK" if trig else
                                       (f"%{prog*100:.0f}" if ic else "")})
        kaynak = self.source_label.get().replace("Kaynak: ", "")
        return {
            "fps": round(st.get("fps", 0), 1),
            "kisi": st.get("n_person", 0),
            "kaynak": kaynak,
            "kaynak_kisa": (kaynak.split("@")[-1].split("/")[0] if "@" in kaynak
                            else kaynak[:22]),
            "duraklat": bool(self.paused.get()),
            "qa": {"ok": st["qa_ok"], "toplam": st["qa_total"], "durum": qa_durum,
                   "alt": qa_alt, "adimlar": adimlar},
            "motor": {"durum": m_state, "kayma": round(st.get("motor_disp", 0), 4),
                      "esik": st.get("motor_thres", 0.1),
                      "gorunmuyor": (0.0 if st.get("motor_present") else
                                     round(st.get("motor_lost_for", 0), 1))},
            "bolgeler": bolgeler,
            "aktif": {"sn": round(st.get("aktif_sn", 0), 1),
                      "islem": round(st.get("aktif_islem", 0), 1)},
            "gun": getattr(self, "_web_gun", {}),
            "son_cevrimler": getattr(self, "_web_cevrimler", []),
            "son_cevrim_id": getattr(self, "_web_cevrim_id", 0),
            "son_cevrim": getattr(self, "_web_son_cevrim", {}),
            "ayarlar": self._web_ayarlar(),
            "izleyici": getattr(self.web, "izleyici", 0),
            "birim": sb(), "birim_kat": (1.0 / SEC_PER_CMIN if BIRIM == "cmin" else 1.0),
        }

    def _ayar_haritasi(self):
        return {"conf_qa": self.conf_qa,
                "conf_kam": self.conf_kam, "conf_musur": self.conf_musur,
                "conf_motor": self.conf_motor, "dwell_kam": self.dwell_kam,
                "dwell_musur": self.dwell_musur, "motor_move": self.motor_move,
                "motor_hold": self.motor_hold, "motor_lost": self.motor_lost,
                "yuz_kabalik": self.yuz_kabalik, "frame_skip": self.frame_skip,
                "alpha": self.alpha, "imgsz_idx": self.imgsz_idx,
                "yuz_bulanik": self.yuz_bulanik, "show_pose": self.show_pose,
                "show_zones": self.show_zones, "show_motor_gate": self.show_motor_gate,
                "rapor_va": self.rapor_va,
                "motor_absent_closes": self.motor_absent_closes}

    def _web_ayarlar(self):
        # her karede 20 tk degiskeni okumak gereksiz: saniyede bir tazele
        kes, zaman = self._web_ayar_kes
        if kes and (time.time() - zaman) < 1.0:
            return kes
        d = {}
        for ad, degisken in self._ayar_haritasi().items():
            try:
                v = degisken.get()
                d[ad] = bool(v) if isinstance(degisken, tk.BooleanVar) else round(float(v), 3)
            except Exception:
                pass
        self._web_ayar_kes = (d, time.time())
        return d

    def web_komut(self, veri):
        """Tarayiciden gelen komut: HTTP thread'inden gelir, KUYRUGA konur.

        Tk arayuzune baska bir thread'den dokunmak guvenli degil; komutlar
        _web_kuyruk_isle() tarafindan Tk dongusunde uygulanir."""
        self.web_q.put(veri or {})

    def _web_kuyruk_isle(self):
        try:
            while True:
                self._web_uygula(self.web_q.get_nowait())
        except queue.Empty:
            pass
        except Exception as e:
            print("[web] komut hatasi:", e)
        if self.alive:
            self.root.after(120, self._web_kuyruk_isle)

    def _web_uygula(self, veri):
        k = (veri or {}).get("komut", "")
        if k == "duraklat":
            self.toggle_pause()
        elif k == "qa_sifirla":
            self.reset_qa()
        elif k == "bolge_sifirla":
            self.reset_zones()
        elif k == "snapshot":
            self.snapshot()
        elif k == "rapor":
            self.rapor()
        elif k == "rapor_ai":
            self.rapor(sunucu=True)
        elif k == "ayar":
            ad = veri.get("ad"); deger = veri.get("deger")
            degisken = self._ayar_haritasi().get(ad)
            if degisken is not None:
                try:
                    if isinstance(degisken, tk.BooleanVar):
                        degisken.set(bool(float(deger)))
                    elif isinstance(degisken, tk.IntVar):
                        degisken.set(int(float(deger)))
                    else:
                        degisken.set(float(deger))
                    self._web_ayar_kes = ({}, 0.0)      # onbellegi tazele
                except Exception as e:
                    print("[web] ayar uygulanamadi:", ad, e)
        else:
            print("[web] bilinmeyen komut:", k)

    # ── uretim kaydi ────────────────────────────────────────────────────
    def _log_cycle(self, cyc, t_end, sonuc, ok, total, eksik):
        """Bir cevrimi CSV'ye yazar (worker thread'inden cagrilir)."""
        row = cevrim_satiri(cyc, t_end, sonuc, ok, total, eksik)
        cevrim = _f(row["cevrim_sn"]); islem = _f(row["va_span_sn"])
        self.logger.append(row)
        self.cycle_rows.append(row)
        durus = " [DURUS]" if cevrim > CEVRIM_MAX_SN else ""
        print(f"[KAYIT]{durus} {row['baslangic']}-{row['bitis']} | "
              f"{sg(cevrim)} {sb()} ({nf(cevrim)} sn) | islem {sg(islem,1)} {sb()} | "
              f"ilk isleme {row['nva_bas_sn']} · arasi {row['nva_ara_sn']} · "
              f"sonrasi {row['nva_son_sn']} sn | {sonuc}")
        self._refresh_cycle_stats()

    def _flush_cycle(self, cyc, t_end, info):
        sonuc, ok, tot, eksik = info or ("", 0, 0, [])
        try:
            self._log_cycle(cyc, t_end, sonuc, ok, tot, eksik)
        except Exception as e:
            print("[kayit] cevrim yazilamadi:", e)

    def _refresh_cycle_stats(self):
        bugun = datetime.now().strftime("%Y-%m-%d")
        rows = [r for r in self.cycle_rows if r.get("tarih") == bugun]
        o = _ozetle(rows)
        oh = _ozetle([r for r in rows if r.get("saat") == datetime.now().strftime("%H")])
        son = _f(rows[-1].get("cevrim_sn")) if rows else 0.0
        T = TEMA
        veri = {
            "son":  (f"{sg(son)} {sb()}", f"{nf(son)} sn", T["cyan"]),
            "ort":  (f"{sg(o['ort_cevrim'])} {sb()}", f"{nf(o['ort_cevrim'])} sn", None),
            "hiz":  (f"{nf(o['parca_saat'],1)}", "parca/saat", None),
            "adet": (f"{o['adet']}", f"bu saat {oh['adet']}", None),
        }
        self.root.after(0, self._kpi_uygula, veri)

        # web paneli icin gun ozeti ve son cevrimler
        self._web_gun = {"adet": o["adet"], "ort": round(o["ort_cevrim"], 1),
                         "hiz": round(o["parca_saat"], 1),
                         "son": round(son, 1), "bu_saat": oh["adet"]}
        self._web_cevrimler = [{"sn": round(_f(r.get("cevrim_sn")), 1),
                                "sonuc": r.get("sonuc", "")} for r in rows[-40:]]
        if rows:
            r = rows[-1]
            self._web_cevrim_id = len(rows)
            self._web_son_cevrim = {"sn": round(_f(r.get("cevrim_sn")), 1),
                                    "sonuc": r.get("sonuc", ""),
                                    "eksik": r.get("eksik", "")}

    def _kpi_uygula(self, veri):
        for anahtar, (deger, altyazi, c) in veri.items():
            if anahtar in getattr(self, "kpi", {}):
                v, a = self.kpi[anahtar]
                v.config(text=deger, fg=c or TEMA["metin"])
                a.config(text=altyazi)

    # ── klip kaydi ──────────────────────────────────────────────────────
    def klip_kaydet(self, etiket="eksik", sn=None, bilgi=None, ek="", aralik=None):
        """Tampondan bir bolumu arka planda mp4 olarak yazar."""
        kareler = (self.clips.aralik(*aralik) if aralik else self.clips.al(sn))
        if not kareler and aralik:
            kareler = self.clips.al(KLIP_MAX_SN)
        if not kareler:
            self.root.after(0, self.rapor_text.set, "Klip icin kare yok.")
            return None
        yol = klip_adi(kareler[-1][0], etiket, ek)
        baslik = (bilgi or {}).get("baslik", "")

        def _job():
            t0 = time.time()
            try:
                pth = klip_yaz(kareler, yol, baslik)
                if pth and bilgi:
                    try:
                        with open(pth[:-4] + ".txt", "w", encoding="utf-8") as f:
                            for k, v in bilgi.items():
                                f.write(f"{k}: {v}\n")
                    except Exception:
                        pass
                if pth:
                    mb = os.path.getsize(pth) / 1024.0 / 1024.0
                    print(f"[KLIP] {os.path.basename(pth)} — {len(kareler)} kare, "
                          f"{mb:.1f} MB, {time.time()-t0:.1f} sn'de yazildi")
                    self.root.after(0, self.rapor_text.set,
                                    f"Klip: {os.path.basename(pth)}")
            except Exception as e:
                print("[klip] yazilamadi:", e)
        threading.Thread(target=_job, daemon=True).start()
        return yol

    def klip_simdi(self):
        self.rapor_text.set("Klip yaziliyor...")
        self.klip_kaydet("elle", sn=KLIP_ELLE_SN,
                         bilgi={"baslik": "ELLE KAYIT",
                                "not": "operatorun istegiyle alindi"})

    # ── rapor ───────────────────────────────────────────────────────────
    def rapor(self, sunucu=False):
        self.rapor_text.set("Rapor olusturuluyor...")
        va = bool(self.rapor_va.get())
        def _job():
            try:
                pth = rapor_olustur(self.logger, sunucu=sunucu, va=va)
                self.root.after(0, self.rapor_text.set, f"Rapor: {os.path.basename(pth)}")
            except Exception as e:
                self.root.after(0, self.rapor_text.set, f"Rapor hatasi: {e}")
        threading.Thread(target=_job, daemon=True).start()

    def web_ac(self):
        if self.web and getattr(self.web, "adres", None):
            try: webbrowser.open(self.web.adres)
            except Exception as e: print("[web] acilamadi:", e)

    def tanitim_ac(self):
        """Uygulamayi anlatan kaydirmali tanitim sayfasi (tanitim/ klasoru)."""
        try:
            if self.web and getattr(self.web, "adres", None):
                webbrowser.open(self.web.adres + "/tanitim/")
            elif os.path.exists(os.path.join(TANITIM_DIR, "index.html")):
                webbrowser.open("file:///" + os.path.join(TANITIM_DIR, "index.html").replace("\\", "/"))
        except Exception as e:
            print("[web] tanitim acilamadi:", e)

    def kayit_klasor(self, yol=None):
        yol = yol or self.logger.dir
        try:
            os.makedirs(yol, exist_ok=True)
        except Exception:
            pass
        try:
            os.startfile(yol)                      # Windows
        except Exception:
            try: webbrowser.open("file:///" + yol.replace("\\", "/"))
            except Exception as e: print("[kayit] klasor acilamadi:", e)

    # ── ROI fare ile duzenleme ─────────────────────────────────────────
    def _canvas_to_norm(self, ex, ey):
        if self.disp_geom is None: return None
        off_x, off_y, scale, fw, fh = self.disp_geom
        fx = (ex - off_x) / max(1e-6, scale)
        fy = (ey - off_y) / max(1e-6, scale)
        return min(1.0, max(0.0, fx / fw)), min(1.0, max(0.0, fy / fh))

    def _on_press(self, ev):
        if not self.edit_roi.get(): return
        self.drag_start = self._canvas_to_norm(ev.x, ev.y)

    def _on_drag(self, ev):
        if not self.edit_roi.get() or self.drag_start is None: return
        cur = self._canvas_to_norm(ev.x, ev.y)
        if cur is None: return
        x1, y1 = self.drag_start; x2, y2 = cur
        self.drag_rect = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))

    def _on_release(self, ev):
        if not self.edit_roi.get() or self.drag_start is None: return
        cur = self._canvas_to_norm(ev.x, ev.y)
        self.drag_start = None
        if cur is None or not self.zones: return
        r = self.drag_rect
        self.drag_rect = None
        if r is None or (r[2]-r[0]) < 0.02 or (r[3]-r[1]) < 0.02:
            return
        with self.zone_lock:
            self.zones[0].rect = r
            self.zones[0].reset()
        print(f"[ROI] {self.zones[0].name} yeni rect = ({r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}, {r[3]:.3f})")

    # ── actions ─────────────────────────────────────────────────────────
    def browse(self):
        path = filedialog.askopenfilename(initialdir=VIDEO_DIR,
                                          filetypes=[("Video","*.mp4 *.mov *.avi *.mkv"),("All","*.*")])
        if path: self.request_source(("video", path), os.path.basename(path))
    def connect_ip(self):
        url = self.ip_url.get().strip()
        if not url: self.ip_status.set("Once bir URL girin."); return
        self.ip_status.set("Baglaniliyor..."); self.request_source(("ip", url), url)
    def request_source(self, sel, label): self.cmd_q.put(("open", sel, label))
    def toggle_pause(self): self.paused.set(not self.paused.get())
    def reset_qa(self): self.qa_reset_evt.set()
    def reset_zones(self):
        with self.zone_lock:
            for z in self.zones: z.reset()
    def snapshot(self):
        if self.last_annotated is not None:
            cv2.imwrite(os.path.join(BASE_DIR, f"snapshot_{int(time.time())}.png"), self.last_annotated)

    # ── capture ─────────────────────────────────────────────────────────
    def _open_capture(self, kind, target):
        if kind == "cam":
            cap = cv2.VideoCapture(target, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        elif kind == "ip":
            cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)
            if cap.isOpened():
                cap = IpOkuyucu(cap)          # surekli okuma: paket kaybi / gri kare olmaz
        else:
            cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG)
            if not cap.isOpened(): cap = cv2.VideoCapture(target, cv2.CAP_MSMF)
        return cap

    def _video_fps(self, cap):
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        return fps if fps and fps == fps and 1 <= fps <= 240 else 30.0

    def _drain_commands(self, cap, kind, vs):
        shutdown = frame_reset = False
        try:
            while True:
                cmd = self.cmd_q.get_nowait()
                if cmd[0] == "open":
                    if cap is not None: cap.release(); time.sleep(0.05)
                    _, sel, label = cmd; kind, target = sel
                    new_cap = self._open_capture(kind, target)
                    if new_cap.isOpened():
                        cap = new_cap
                        self.root.after(0, self.source_label.set, f"Kaynak: {label}")
                        if kind != "ip": self.root.after(0, self.ip_status.set, "")
                        frame_reset = True
                        if kind == "video":
                            vs = VideoState(fps=self._video_fps(cap), start_wall=time.time())
                    else:
                        cap = None
                        self.root.after(0, self.source_label.set, f"Basarisiz: {label}")
                elif cmd[0] == "stop":
                    if cap is not None: cap.release(); cap = None
                    self.root.after(0, self.source_label.set, "(durduruldu)")
                elif cmd[0] == "shutdown":
                    if cap is not None: cap.release(); cap = None
                    shutdown = True; break
        except queue.Empty: pass
        return cap, kind, vs, shutdown, frame_reset

    def _read_frame(self, cap, kind, vs, last_frame):
        ok, frame = cap.read()
        if not ok:
            if kind == "video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); vs.start_wall = time.time(); vs.start_frame = 0
                return None, cap, vs
            if kind == "ip": return None, cap, vs
            cap.release(); self.root.after(0, self.source_label.set, "(durduruldu)")
            return None, None, vs
        if kind == "video" and vs.start_wall is not None:
            elapsed = time.time() - vs.start_wall
            target = vs.start_frame + int(elapsed * vs.fps)
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES)); drops = 0
            while cur < target and drops < 30:
                ok2, f2 = cap.read()
                if not ok2: break
                frame, cur, drops = f2, cur+1, drops+1
        return frame, cap, vs

    def _musur_model_tespit(self, frame, imgsz):
        """Ayri musur modelinin tespitleri: [(ad, cx, cy, (x1,y1,x2,y2) normalize, conf, piksel_kutu)]."""
        try:
            r = self.model_musur.predict(source=frame, **self._pk(self.ov_musur, imgsz, MUSUR_MODEL_CONF))[0]
        except Exception as e:
            print("musur model hata:", e); return []
        if r.boxes is None or not len(r.boxes):
            return []
        fh, fw = frame.shape[:2]
        out = []
        for box in r.boxes:
            ad = MUSUR_MODEL_SINIF.get(self.model_musur.names[int(box.cls[0])])
            if ad is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            out.append((ad, (x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh,
                        (x1 / fw, y1 / fh, x2 / fw, y2 / fh), float(box.conf[0]),
                        (int(x1), int(y1), int(x2), int(y2))))
        return out

    # ── hibrit cikarim ──────────────────────────────────────────────────
    def _infer_all(self, frame, conf_qa, per_conf, imgsz, alpha):
        now_t = time.time()

        # ── KVKK: kisiler EN ALT katmanda bulaniklastirilir ──────────
        # Poz modeli, iskelet gosterimi kapali olsa bile maskeleme icin calisir.
        rp = None
        if self.model_pose is not None and (self.show_pose.get() or self.yuz_bulanik.get()):
            try:
                rp = self.model_pose.predict(source=frame, **self._pk(self.ov_pose, imgsz, POSE_CONF))[0]
            except Exception as e:
                print("pose hata:", e)
        base = frame
        if self.yuz_bulanik.get():
            fh0, fw0 = frame.shape[:2]
            kisiler, baslar, siluet = [], [], None
            if KISI_BULANIK and getattr(self, "model_seg", None) is not None:
                try:
                    rs = self.model_seg.predict(source=frame, classes=[0],
                                                **self._pk(self.ov_seg, KISI_SEG_IMGSZ, KISI_SEG_CONF))[0]
                    if rs.masks is not None and len(rs.masks):
                        poz_kut = (kisi_kutulari(rp.boxes, fw0, fh0, pay=KISI_KIRP)
                                   if rp is not None else [])
                        koridor = kisi_koridor(rp.keypoints, fw0, fh0) if rp is not None else None
                        siluet = np.zeros((fh0, fw0), np.uint8)
                        kxy = rs.boxes.xyxy.cpu().numpy(); kcf = rs.boxes.conf.cpu().numpy()
                        for poli, (sx1, sy1, sx2, sy2), cf in zip(rs.masks.xy, kxy, kcf):
                            if len(poli) < 3:
                                continue
                            alan = max(1.0, (sx2 - sx1) * (sy2 - sy1))
                            eslesen = None
                            for px1, py1, px2, py2 in poz_kut:
                                ort = (max(0.0, min(sx2, px2) - max(sx1, px1))
                                       * max(0.0, min(sy2, py2) - max(sy1, py1))) / alan
                                if ort >= KISI_ESLES:
                                    eslesen = (px1, py1, px2, py2); break
                            if poz_kut and eslesen is None:
                                continue                    # poz kisi gormedi -> insan degil
                            if not poz_kut and cf < KISI_SEG_TEK_CONF:
                                continue
                            tek = np.zeros_like(siluet)
                            cv2.fillPoly(tek, [poli.astype(np.int32)], 255)
                            if eslesen is not None:          # kisinin kutusu disina tasma
                                px1, py1, px2, py2 = eslesen
                                tek[:py1, :] = 0; tek[py2:, :] = 0; tek[:, :px1] = 0; tek[:, px2:] = 0
                            if koridor is not None:
                                tek &= koridor               # motor siluete karistiysa ayikla
                            siluet |= tek
                        if not siluet.any():
                            siluet = None
                except Exception as e:
                    print("seg hata:", e)
            if rp is not None:
                baslar = bas_bolgeleri(rp.keypoints, fw0, fh0)   # bas siluetten tassa da kapanir
                if KISI_BULANIK and siluet is None:
                    kisiler = kisi_kutulari(rp.boxes, fw0, fh0)  # yedek: siluet yoksa kutu
            son = getattr(self, "_maske_son", None)
            if kisiler or baslar or siluet is not None:
                self._maske_son = (now_t, kisiler, baslar, siluet)
            elif son and (now_t - son[0]) <= YUZ_HATIRLA:
                kisiler, baslar, siluet = son[1], son[2], son[3]   # kisa kayiplarda son konum
            if kisiler or baslar or siluet is not None:
                base = frame.copy()
                kab = int(self.yuz_kabalik.get())
                if siluet is not None:
                    base = kisi_siluet_bulanik(base, siluet, kab)
                if kisiler:
                    base = kisi_bulanik(base, kisiler, kab)
                if baslar:
                    base = yuz_mozaik(base, baslar, kab)
        layer = base.copy()
        steps_seen = []; work_seen = []; n_qa = 0
        motor_cands = []      # [(alan, (cx, cy)), ...] — bu karedeki motor kutulari
        kep_boxes_norm = []   # [(cx, cy)] normalize — kep_sokme tespitleri
        musur_boxes = []      # [(ad, cx, cy)] normalize — musur sikma/takma tespitleri
        self._musur_boxes = musur_boxes
        kam_boxes = []        # [(cx, cy)] normalize — 'kam_mili' nesnesi
        self._kam_boxes = kam_boxes

        # QA modeli: islem siniflari + motor (her birine ayri confidence)
        if self.model_qa is not None:
            # en dusuk esikle tahmin al, sonra her sinifi kendi esigiyle suz
            floor = min([conf_qa] + list(per_conf.values()))
            rk = self.model_qa.predict(source=frame, **self._pk(self.ov_qa, imgsz, floor))[0]
            names = self.model_qa.names
            keep = []
            if rk.boxes is not None:
                fh, fw = frame.shape[:2]
                for i, box in enumerate(rk.boxes):
                    name = names[int(box.cls[0])]
                    thr = per_conf.get(name, conf_qa)   # sinifa ozel yoksa genel
                    if float(box.conf[0]) < thr:
                        continue
                    if name in GIZLI_CLASSES:
                        continue          # KKD siniflari: cizilmez, sayilmaz
                    keep.append(i)
                    x1b, y1b, x2b, y2b = [float(v) for v in box.xyxy[0]]
                    if name == MOTOR_CLASS:
                        area = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)
                        motor_cands.append((area, ((x1b + x2b) / 2.0 / fw,
                                                   (y1b + y2b) / 2.0 / fh),
                                            (x1b / fw, y1b / fh, x2b / fw, y2b / fh)))
                    elif name == KEP_SINIF:
                        # KEP kutu merkezini normalize koordinat olarak sakla
                        kep_boxes_norm.append(((x1b + x2b) / 2.0 / fw,
                                               (y1b + y2b) / 2.0 / fh))
                    elif name == KAM_NESNE:
                        kam_boxes.append(((x1b + x2b) / 2.0 / fw, (y1b + y2b) / 2.0 / fh))
                    elif name in MUSUR_KONUM_SINIFLAR:
                        musur_boxes.append((name, (x1b + x2b) / 2.0 / fw,
                                            (y1b + y2b) / 2.0 / fh,
                                            (x1b / fw, y1b / fh, x2b / fw, y2b / fh)))
                    # suresi olculen is (QA adimi olsun olmasin) analize girer
                    if name in IS_WORK_SET: work_seen.append(name)
                    if name in IGNORE_CLASSES: continue
                    if name in QA_STEP_CLASSES: steps_seen.append(name)
                n_qa = len(keep)
                if len(keep) < len(rk.boxes):
                    rk = rk[keep]
            # motor kutusu ayri renkte (MOTOR_RENK) cizilir, digerleri Ultralytics paletiyle
            motor_idx, diger_idx = [], []
            if rk.boxes is not None:
                for i, box in enumerate(rk.boxes):
                    (motor_idx if names[int(box.cls[0])] == MOTOR_CLASS else diger_idx).append(i)
            if diger_idx:
                layer = (rk if not motor_idx else rk[diger_idx]).plot(img=layer, boxes=True)
            # Bazi Ultralytics surumleri (or. 8.4.x) plot() ciktisini SALT OKUNUR
            # dondurur; ustune cizebilmek icin yazilabilir kopya al.
            if not layer.flags.writeable:
                layer = layer.copy()
            for i in motor_idx:
                box = rk.boxes[i]
                x1m, y1m, x2m, y2m = [int(v) for v in box.xyxy[0]]
                lw = max(2, int(round(sum(layer.shape[:2]) / 2 * 0.003)))
                cv2.rectangle(layer, (x1m, y1m), (x2m, y2m), MOTOR_RENK, lw, cv2.LINE_AA)
                etk = f"{MOTOR_CLASS} {float(box.conf[0]):.2f}"
                fs = max(0.5, lw / 3)
                (tw, th), bl = cv2.getTextSize(etk, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, lw - 1))
                ty = y1m - 4 if y1m - th - 8 > 0 else y1m + th + 6
                cv2.rectangle(layer, (x1m, ty - th - 4), (x1m + tw + 6, ty + bl), MOTOR_RENK, -1)
                cv2.putText(layer, etk, (x1m + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                            (255, 255, 255), max(1, lw - 1), cv2.LINE_AA)
        # NOT: saydam katman HAM kareyle degil, bulanik tabanla karistirilir
        annotated = cv2.addWeighted(base, 1-alpha, layer, alpha, 0) if alpha != 1.0 else layer

        hand_points = []
        n_person = 0
        if rp is not None:
            n_person = len(rp.boxes) if rp.boxes is not None else 0
            if self.show_pose.get():
                if not annotated.flags.writeable:
                    annotated = annotated.copy()
                annotated, wrists = draw_pose(annotated, rp.keypoints)   # bulanigin USTUNE
                hand_points.extend(wrists)

        return (annotated, n_qa, steps_seen, work_seen, hand_points,
                n_person, motor_cands, kep_boxes_norm)

    # ── worker ──────────────────────────────────────────────────────────
    def _worker_main(self):
        cap = kind = ip_target = None
        ip_fail_streak = 0
        last_frame = last_annotated = None
        last_n = skip_counter = 0
        t_prev = time.time(); fps_smoothed = 0.0
        vs = VideoState()
        qa = QAChecklist(QA_STEP_CLASSES); last_stats = None
        step_confirm = StepConfirm()
        motor_mon = MotorCycleMonitor()
        kep_tracker = KepSokmeTracker(motor_mon, n=KEP_COUNT)
        kep_yedek = None    # yanlis kapanmada geri yuklenecek kep durumu
        musur_gorulen = 0   # bu cevrimde musurun yerinde goruldugu kare sayisi
        musur_poz = None    # musurun bu cevrimde en son goruldugu ekran konumu
        musur_tani_t = 0.0  # son tani mesajinin zamani
        ertelenen = {}      # sirasi gelmeden gorulen adimlar {ad: zaman}
        gec_bekliyor = False; gec_bitis = 0.0; gec_ok = gec_tot = 0
        son_kapanan_cyc = None
        pre_steps = {}      # cevrim oncesi yapilan adimlar {ad: zaman}
        pre_block_until = 0.0   # kapanis sonrasi kisa sure kayit alma
        cyc = None              # aktif cevrimin sure birikimi
        cyc_pending = None      # kapandi ama geri alinabilir (henuz yazilmadi)
        cyc_close_t = 0.0
        pend_info = None        # (sonuc, ok, toplam, eksik) kapanis anindaki durum

        def _yeni_cevrim(t):
            # Cevrim, onceki parca ciktigi anda baslar; motor yerlesmesini
            # BEKLEMEZ — gelirken yapilan islemler de sayilir.
            return dict(t0=t, last_t=t, dur=0.0, va=0.0, arm_dur=None, spans={},
                        step_sec={s: 0.0 for s in IS_CLASSES}, done_at={})

        cyc = _yeni_cevrim(time.time())   # ilk cevrim program acilisinda basliyor
        zone_by_name = {z.name: z for z in self.zones}
        prev_done = {s: False for s in QA_STEP_CLASSES}

        def _blocked(name):
            """Adim su an isaretlenebilir mi? Engel varsa sebebini dondurur."""
            for r in STEP_REQUIRES.get(name, ()):
                if not qa.done.get(r, False):
                    return f"once {pretty(r)}"
            herhangi = STEP_REQUIRES_ANY.get(name)
            if herhangi and not any(qa.done.get(r, False) for r in herhangi):
                return "once " + " veya ".join(pretty(r) for r in herhangi)
            zn = STEP_ZONE_GATE.get(name)
            if zn:
                z = zone_by_name.get(zn)
                if z is not None and not z.triggered:
                    return f"{zn} bekleniyor"
            return None
        pass_since = None   # sonuc gosterim sayaci
        dbg_t = time.time()
        closed_result = None      # "GECTI" | "EKSIK"
        closed_missing = []

        def _full_reset(reason=""):
            qa.reset()
            ertelenen.clear()
            step_confirm.reset()
            # NOT: kep sayaci burada SIFIRLANMAZ. Kepler motor gelirken de
            # sokulebiliyor; sayac motor CIKINCA (close) ve elle sifirlamada sifirlanir.
            with self.zone_lock:
                for z in self.zones: z.reset()
            if reason:
                print(f"[QA] sifirlandi — {reason}")

        while self.alive:
            cap, kind, vs, shutdown, frame_reset = self._drain_commands(cap, kind, vs)
            if shutdown: return
            if frame_reset:
                last_frame = None; ip_fail_streak = 0
                if kind == "ip": ip_target = self.ip_url.get().strip()
            if self.qa_reset_evt.is_set():
                _full_reset()
                kep_tracker.reset()
                musur_gorulen = 0; musur_poz = None; gec_bekliyor = False
                motor_mon.reset()
                cyc = _yeni_cevrim(time.time())
                cyc_pending = None
                pre_steps.clear()
                pass_since = None; closed_result = None; closed_missing = []
                self.qa_reset_evt.clear()
            if cap is None: time.sleep(0.05); continue

            if not self.paused.get():
                frame, cap, vs = self._read_frame(cap, kind, vs, last_frame)
                if kind == "ip" and frame is None and cap is not None:
                    ip_fail_streak += 1
                    if ip_fail_streak <= IP_RECONNECT_ATTEMPTS:
                        self.root.after(0, self.ip_status.set,
                                        f"Yeniden baglaniliyor ({ip_fail_streak}/{IP_RECONNECT_ATTEMPTS})...")
                        time.sleep(IP_RECONNECT_DELAY); cap.release()
                        cap = self._open_capture("ip", ip_target)
                        if not cap.isOpened(): cap = None
                    else:
                        self.root.after(0, self.ip_status.set, "Basarisiz — URL kontrol.")
                        self.root.after(0, self.source_label.set, "(durduruldu)")
                        cap.release(); cap = None; ip_fail_streak = 0
                    continue
                if frame is not None:
                    ip_fail_streak = 0
                    if kind == "ip": self.root.after(0, self.ip_status.set, "Bagli")
                if cap is None or frame is None: continue
                last_frame = frame
            else:
                if last_frame is None: time.sleep(0.03); continue
                frame = last_frame

            conf_qa = float(self.conf_qa.get())
            per_conf = self._live_conf_map()
            dwell_map = self._live_dwell_map()
            move_thres = float(self.motor_move.get())
            imgsz = IMGSZ_CHOICES[int(self.imgsz_idx.get())]
            alpha = float(self.alpha.get()); skip_n = int(self.frame_skip.get())

            if skip_n > 0 and skip_counter < skip_n and last_annotated is not None:
                skip_counter += 1
                if last_stats is not None:
                    self.root.after(0, self._update_display, last_annotated, last_stats)
                continue
            skip_counter = 0

            try:
                (last_annotated, n_qa, steps_seen, work_seen, hand_points,
                 n_person, motor_cands, kep_boxes_norm) = self._infer_all(
                    frame, conf_qa, per_conf, imgsz, alpha)
                self.last_annotated = last_annotated
                last_n = n_qa
            except Exception as e:
                print("infer hata:", e); continue

            now = time.time()
            fps_smoothed = 0.9*fps_smoothed + 0.1/max(1e-6, now - t_prev); t_prev = now

            # ── KEP SOKME + MUSUR (v8) ────────────────────────────────
            # takip edilen motorun kutusu = son konuma en yakin aday
            motor_box = None
            if motor_cands:
                ref = motor_mon.smooth or motor_mon.anchor
                best = (min(motor_cands, key=lambda c: _motor_dist(c[1], ref)) if ref
                        else max(motor_cands, key=lambda c: c[0]))
                motor_box = best[2]
            # support motor paletine bagli: bolgeyi motorla birlikte tasi
            with self.zone_lock:
                supp_zone = next((z for z in self.zones if z.name == "Kep support"), None)
                if supp_zone is not None and (motor_box or kep_tracker.motor_box):
                    supp_zone.rect = kep_tracker.support_kutusu()
                support_aktif = bool(supp_zone and supp_zone.triggered)
            # Sayim yalnizca motor postaya YERLESMISKEN yapilir: cikan motorun
            # dolu support'u yeni motora yazilmasin.
            sayim_acik = motor_mon.armed and now >= pre_block_until
            if sayim_acik:
                kep_tracker.feed(now, kep_boxes_norm, motor_box, support_aktif)
            # kep_sokme adimini YALNIZCA sayac isaretler (tek etiket yetmez)
            steps_seen = [s for s in steps_seen if s != "kep_sokme"]
            if kep_tracker.qa_gecti_mi() and not qa.done.get("kep_sokme", False):
                qa.mark("kep_sokme", "model")

            # ── MUSUR (istege bagli ayri model) ───────────────────────
            if (getattr(self, "model_musur", None) is not None and sayim_acik
                    and not (qa.done.get(MUSUR_TAKILI_ADIM, False) and qa.done.get("musur_skma", False))):
                ek = self._musur_model_tespit(frame, imgsz)
                if ek:
                    self._musur_boxes = list(getattr(self, "_musur_boxes", [])) + [
                        (ad, cx, cy, k) for ad, cx, cy, k, _c, _p in ek]
                    if any(e_[0] == "musur_skma" for e_ in ek) and "musur_skma" not in steps_seen:
                        steps_seen.append("musur_skma")      # konum kontrolu asagida yapilir
                    if not last_annotated.flags.writeable:
                        last_annotated = last_annotated.copy()
                    for ad, _cx, _cy, _k, cf, (x1, y1, x2, y2) in ek:
                        renk = MUSUR_MODEL_RENK.get(ad, (200, 200, 200))
                        cv2.rectangle(last_annotated, (x1, y1), (x2, y2), renk, 2, cv2.LINE_AA)
                        cv2.putText(last_annotated, f"{'musur' if ad == MUSUR_NESNE else 'musur sikma'} {cf:.2f}",
                                    (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, renk, 2, cv2.LINE_AA)
                    self.last_annotated = last_annotated

            mbox = kep_tracker.motor_box or motor_box
            musur_kutular = getattr(self, "_musur_boxes", [])
            # 'müşür' NESNESI: motor govdesinin alt kisminda goruldu mu?
            # Referans kutusu VEYA o anki motor kutusu kabul edilir (canli kamerada
            # biri kirpilmis/kaymis olabilir). Sabit noktaya bakilmaz -> motor
            # palette dondurulse de calisir.
            musur_tespit = [(cx, cy) for ad, cx, cy, _k in musur_kutular if ad == MUSUR_NESNE]
            neden = None
            if musur_tespit:
                kutular = [b_ for b_ in (kep_tracker.motor_box, motor_box) if b_ is not None]
                yerinde = None
                for cx, cy in musur_tespit:
                    for bx1, by1, bx2, by2 in kutular:
                        ry_ = (cy - by1) / max(1e-6, by2 - by1)
                        if (bx1 - MUSUR_X_PAY <= cx <= bx2 + MUSUR_X_PAY
                                and MUSUR_NESNE_RY[0] <= ry_ <= MUSUR_NESNE_RY[1]):
                            yerinde = (cx, cy); break
                    if yerinde: break
                if not kutular:
                    neden = "motor kutusu yok"
                elif yerinde is None:
                    bx1, by1, bx2, by2 = kutular[0]; cx, cy = musur_tespit[0]
                    neden = (f"motor bolgesi disinda (x={cx:.2f}, kutu x {bx1:.2f}-{bx2:.2f}; "
                             f"ry={(cy - by1) / max(1e-6, by2 - by1):.2f}, beklenen "
                             f"{MUSUR_NESNE_RY[0]:.2f}-{MUSUR_NESNE_RY[1]:.2f})")
                elif not sayim_acik:
                    neden = f"motor POSTADA degil (durum={motor_mon.state})"
                else:
                    musur_poz = yerinde
                    musur_gorulen += 1
                # tani: musur goruluyor ama sayilmiyorsa nedeni 3 sn'de bir yaz
                if neden and now - musur_tani_t > 3.0 and not qa.done.get(MUSUR_TAKILI_ADIM, False):
                    print(f"[MUSUR] tespit var ama sayilmadi: {neden}")
                    musur_tani_t = now
            if sayim_acik and musur_gorulen >= MUSUR_MIN_KARE and not qa.done.get(MUSUR_TAKILI_ADIM, False):
                qa.mark(MUSUR_TAKILI_ADIM, "model")
                print(f"[MUSUR] musur goruldu ({musur_gorulen} kare) -> TAKILI")

            def _musur_sikma_yerinde():
                """musur_skma kutusu: musur noktasini (harita ya da gercek konum)
                iceriyor VEYA motor govdesinin alt kismiyla yeterince ortusuyor.
                v8: referans kutusu VE o anki motor kutusu ayri ayri denenir
                (motor palette dondurulunce referans eskiyebiliyor)."""
                kutular_ = [b_ for b_ in (kep_tracker.motor_box, motor_box) if b_ is not None]
                p_ = MUSUR_SKMA_PAD
                for mx1, my1, mx2, my2 in kutular_:
                    noktalar = [(mx1 + MUSUR_KONUM[0] * (mx2 - mx1), my1 + MUSUR_KONUM[1] * (my2 - my1))]
                    if musur_poz is not None:
                        noktalar.append(musur_poz)
                    ly1 = my1 + MUSUR_ALT_RY[0] * (my2 - my1)
                    for ad, _, _, k in musur_kutular:
                        if ad != "musur_skma":
                            continue
                        if any(k[0] - p_ <= px <= k[2] + p_ and k[1] - p_ <= py <= k[3] + p_ for px, py in noktalar):
                            return True
                        ix = max(0.0, min(k[2], mx2) - max(k[0], mx1))
                        iy = max(0.0, min(k[3], my2) - max(k[1], ly1))
                        alan = max(1e-9, (k[2] - k[0]) * (k[3] - k[1]))
                        if ix * iy / alan >= MUSUR_SKMA_MIN_ORT:
                            return True
                return False
            # musur_skma ISLEMI musur bolgesinde degilse sayilmaz
            skma_neden = None
            if MUSUR_KONUM_KONTROL and mbox is not None and "musur_skma" in steps_seen:
                if not _musur_sikma_yerinde():
                    steps_seen = [s for s in steps_seen if s != "musur_skma"]
                    k = next(k for ad, _, _, k in musur_kutular if ad == "musur_skma")
                    skma_neden = (f"konum disi (kutu x {k[0]:.2f}-{k[2]:.2f} y {k[1]:.2f}-{k[3]:.2f}; "
                                  f"motor x {mbox[0]:.2f}-{mbox[2]:.2f} y {mbox[1]:.2f}-{mbox[3]:.2f})")

            # ── KAM MILI NESNESI -> 'kam mili koyma' ──────────────────
            if (KAM_NESNE_SAY and sayim_acik and "kam_mili koyma" in QA_STEP_CLASSES
                    and not qa.done.get("kam_mili koyma", False)
                    and "kam_mili koyma" not in steps_seen):
                kutular_ = [b_ for b_ in (kep_tracker.motor_box, motor_box) if b_ is not None]
                for kcx, kcy in getattr(self, "_kam_boxes", []):
                    if any(bx1 - KAM_X_PAY <= kcx <= bx2 + KAM_X_PAY
                           and kcy >= by1 + KAM_MIN_RY * (by2 - by1)
                           for bx1, by1, bx2, by2 in kutular_):
                        steps_seen.append("kam_mili koyma")
                        break

            # ── SILIKON (istege bagli spatula modeli) ─────────────────
            # Yalnizca silikon beklenirken calisir; spatula motor kutusuna
            # degiyorsa bu kare 'silikon_silme' goruldu sayilir.
            if (self.model_sil is not None and mbox is not None
                    and not qa.done.get("silikon_silme", False)
                    and (motor_mon.armed or gec_bekliyor)
                    and all(qa.done.get(r, False) for r in STEP_REQUIRES.get("silikon_silme", ()))
                    and (not STEP_REQUIRES_ANY.get("silikon_silme")
                         or any(qa.done.get(r, False) for r in STEP_REQUIRES_ANY["silikon_silme"]))):
                try:
                    rs = self.model_sil.predict(source=frame, **self._pk(self.ov_sil, imgsz, SILIKON_CONF))[0]
                except Exception as e:
                    rs = None; print("silikon model hata:", e)
                if rs is not None and rs.boxes is not None and len(rs.boxes):
                    fh_, fw_ = frame.shape[:2]
                    mx1, my1, mx2, my2 = mbox
                    p_ = SILIKON_PAY
                    if not last_annotated.flags.writeable:
                        last_annotated = last_annotated.copy()
                    for box in rs.boxes:
                        ad = self.model_sil.names[int(box.cls[0])]
                        if ad not in SILIKON_SINIFLAR:
                            continue
                        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                        kx1, ky1, kx2, ky2 = x1 / fw_, y1 / fh_, x2 / fw_, y2 / fh_
                        degiyor = (kx2 >= mx1 - p_ and kx1 <= mx2 + p_
                                   and ky2 >= my1 - p_ and ky1 <= my2 + p_)
                        renk = SILIKON_RENK if degiyor else (150, 150, 150)
                        cv2.rectangle(last_annotated, (int(x1), int(y1)), (int(x2), int(y2)), renk, 2, cv2.LINE_AA)
                        cv2.putText(last_annotated, f"{ad} {float(box.conf[0]):.2f}", (int(x1), max(14, int(y1) - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, renk, 2, cv2.LINE_AA)
                        if degiyor:
                            tem = getattr(self, "_sp_temas", None)
                            if tem is None:
                                tem = self._sp_temas = collections.deque()
                            if not tem or tem[-1] != now:
                                tem.append(now)
                            while tem and now - tem[0] > SILIKON_PENCERE:
                                tem.popleft()
                            if len(tem) >= SILIKON_MIN_KARE and not qa.done.get("silikon_silme", False):
                                qa.mark("silikon_silme", "model")   # sart kendi basina yeterli
                                print(f"[SILIKON] spatula {len(tem)} karede motora degdi -> silikon silme")
                    self.last_annotated = last_annotated

            # ── yeni biten adimlar: kapi bolgelerini HEMEN sifirla ──────
            # (kilitler hesaplanmadan once yapilmali; aksi halde onkosul
            #  biten karede etiket bir kare boyunca serbest kalir)
            with self.zone_lock:
                for s in QA_STEP_CLASSES:
                    if qa.done.get(s, False) and not prev_done.get(s, False):
                        if cyc is not None:
                            cyc["done_at"][s] = round(cyc["dur"], 2)
                        for gated, reqs in STEP_REQUIRES.items():
                            if s in reqs:
                                zn = STEP_ZONE_GATE.get(gated)
                                z = zone_by_name.get(zn)
                                if z is not None and z.triggered:
                                    z.reset(warmup=0.0)
                                    print(f"[BOLGE] '{zn}' kapisi sifirlandi — "
                                          f"{pretty(s)} sonrasi tekrar tetiklenmeli")
                prev_done = dict(qa.done)

            # ── adim kilitleri: sirasi gelmemis adim isaretlenmez ────────
            qa_wait = {}
            for name in QA_STEP_CLASSES:
                if not qa.done.get(name, False):
                    b = _blocked(name)
                    if b:
                        qa_wait[name] = b

            # tani: musur sikma goruluyor ama sayilmiyorsa nedeni 3 sn'de bir yaz
            if not qa.done.get("musur_skma", False):
                if skma_neden is None and "musur_skma" in steps_seen and "musur_skma" in qa_wait:
                    if "musur_skma" in ertelenen:
                        skma_neden = None   # bekletiliyor, kep bitince sayilacak
                    elif not motor_mon.armed:
                        skma_neden = f"{qa_wait['musur_skma']} + motor POSTADA degil (durum={motor_mon.state})"
                    else:
                        skma_neden = f"{qa_wait['musur_skma']} (ertelendi, kep bitince sayilacak)"
                if skma_neden and now - getattr(self, "_skma_tani_t", 0.0) > 3.0:
                    print(f"[MUSUR SIKMA] tespit var ama sayilmadi: {skma_neden} | "
                          f"kep {kep_tracker.support_n}/{kep_tracker.n}")
                    self._skma_tani_t = now

            # ertelenmis adimlar: sarti simdi saglandiysa say
            for name in list(ertelenen):
                if qa.done.get(name, False):
                    ertelenen.pop(name, None)
                elif name not in qa_wait and motor_mon.armed:
                    qa.mark(name, "model")
                    print(f"[QA] '{pretty(name)}' sirasi gelmeden {now - ertelenen[name]:.1f} sn "
                          f"once gorulmustu -> simdi sayildi")
                    ertelenen.pop(name, None)

            # ── model tespitinden gelen adimlar (onay suresi/dwell ile) ──
            # kilitli adimda sure ISLEMEZ: etiket kapi acilmadan once
            # gorulmus olsa bile sayilmaz, sayac kapidan sonra baslar.
            # (ERTELENEBILIR adimlar haric: onlar kilitliyken de olculur ve
            #  hazir olunca 'ertelenen'e yazilir)
            for name in steps_seen:
                if name not in qa_wait or (name in ERTELENEBILIR and motor_mon.armed):
                    step_confirm.feed(now, name)
                    if (name in qa_wait and name not in ertelenen
                            and step_confirm.ready(now, name, float(dwell_map.get(name, 0.0)))):
                        ertelenen[name] = now
            qa_prog = {}
            for name in set(steps_seen):
                if name in qa_wait:
                    continue
                d = float(dwell_map.get(name, 0.0))
                if step_confirm.ready(now, name, d):
                    qa.mark(name, "model")
                    # cevrim baslamadan yapilan adimlar (or. motor gelirken kam
                    # mili / kep sokme): yeni cevrime tasinmak uzere kaydet
                    if (not motor_mon.armed and name in PRE_CYCLE_STEPS
                            and now >= pre_block_until):
                        if name not in pre_steps:
                            print(f"[QA] '{pretty(name)}' cevrim oncesi yapildi — "
                                  f"yeni cevrime tasinacak")
                        pre_steps[name] = now
                elif not qa.done.get(name, False):
                    qa_prog[name] = step_confirm.progress(now, name, d)

            # kep sokme ilerlemesi (panelde %): support'taki kep orani
            if not qa.done.get("kep_sokme", False) and kep_tracker.support_n > 0:
                qa_prog["kep_sokme"] = min(1.0, kep_tracker.support_n / kep_tracker.n)

            # BOLGE: el bolgede VEYA bolgede hareket -> bagli adimi isaretle
            fh, fw = frame.shape[:2]
            zone_state = {}
            with self.zone_lock:
                for z in self.zones:
                    fired = z.update(now, hand_points, frame)
                    if fired and z.qa_step:
                        if z.action == "gate":
                            # bolge adimi isaretlemez, sadece kapiyi acar
                            print(f"[BOLGE] '{z.name}' kapisi acildi -> "
                                  f"'{pretty(z.qa_step)}' etiketi bekleniyor")
                        else:
                            qa.mark(z.qa_step, "bolge")
                            print(f"[BOLGE] '{z.name}' tetiklendi -> QA: {z.qa_step}")
                    # kapi acik ama etiket hic gelmiyorsa (opsiyonel yedek)
                    if (z.action == "gate" and z.triggered and z.qa_step
                            and GATE_FALLBACK_SECS > 0 and z.last_trigger_t is not None
                            and not qa.done.get(z.qa_step, False)
                            and _blocked(z.qa_step) is None
                            and (now - z.last_trigger_t) >= GATE_FALLBACK_SECS):
                        qa.mark(z.qa_step, "bolge")
                        print(f"[BOLGE] '{pretty(z.qa_step)}' etiket gelmedi, "
                              f"{GATE_FALLBACK_SECS:.0f} sn sonra bolgeden isaretlendi")
                    zone_state[z.name] = (z.hand_inside or z.motion_active, z.triggered, z.progress)


            # ── MOTOR CEVRIMI: yerlesme / cikis ─────────────────────────
            motor_evt = motor_mon.update(now, motor_cands, move_thres,
                                         float(self.motor_hold.get()),
                                         bool(self.motor_absent_closes.get()),
                                         float(self.motor_lost.get()))

            # ── URETIM SURESI BIRIKIMI (ISLEM / BOS) ────────────────────
            # Motorun yerlesmesi BEKLENMEZ: sayac onceki parca ciktigi anda
            # basladi, motor gelirken yapilan islemler de sayilir.
            if cyc is not None:
                dt_acc = min(ACC_DT_MAX, max(0.0, now - cyc["last_t"]))
                cyc["last_t"] = now
                if not self.paused.get():
                    cyc["dur"] += dt_acc
                    if work_seen:
                        cyc["va"] += dt_acc      # aktif islem sayaci
                        for _s in set(work_seen):
                            span_feed(cyc, _s, now)   # "iki etiket arasi" olcum
                            if _s in cyc["step_sec"]:
                                cyc["step_sec"][_s] += dt_acc

            if motor_evt == "close" and pass_since is None:
                closed_missing = qa.missing()
                closed_result = "GECTI" if not closed_missing else "EKSIK"
                pass_since = now
                # biten parcanin artik etiketleri yeni parcaya yazilmasin
                pre_steps.clear()
                pre_block_until = now + PRE_AFTER_CLOSE_IGNORE
                # kep sayaci yeni motor icin sifirlanir (yanlis kapanmaya karsi yedekle)
                kep_yedek = (copy.copy(kep_tracker.__dict__), musur_gorulen, musur_poz)
                if not kep_tracker.tamamlandi_mi:
                    eks = kep_tracker.eksik_kepler(now)
                    print(f"[KEP] motor cikti, kep sokme EKSIK: {kep_tracker.durum_metni()}"
                          + (f" — motorda kalan kep no: {eks}" if eks else ""))
                    self.son_eksik_kepler = eks
                else:
                    self.son_eksik_kepler = []
                kep_tracker.reset()
                musur_gorulen = 0; musur_poz = None
                # cevrim kaydi: geri alma penceresi gecmeden yazma
                son_kapanan_cyc = cyc
                if cyc is not None:
                    cyc_pending = cyc
                    cyc_close_t = now
                    pend_info = (closed_result, qa.ok_count, qa.total,
                                 list(closed_missing))
                    cyc = _yeni_cevrim(now)   # yeni cevrim HEMEN basliyor
                # Sonuc, kayit ve EKSIK klibi GEC_ADIM_SECS boyunca bekletilir:
                # bu surede gorulen eksik adim biten cevrime yazilir. Kapanma
                # ani / cevrim suresi DEGISMEZ.
                gec_bekliyor = True; gec_bitis = now + GEC_ADIM_SECS
                gec_ok = qa.ok_count; gec_tot = qa.total
            elif motor_evt == "arm":
                # yeni motor postaya yerlesti: sonuc ekranda bekliyorsa hemen sifirla
                if pass_since is not None:
                    pass_since = None; closed_result = None; closed_missing = []
                    _full_reset("yeni motor yerlesti")
                else:
                    _full_reset("yeni motor yerlesti (cevrim basladi)")
                # bekleyen kayit varsa kapanis gercekti -> yaz
                # (gec adim penceresi aciksa once ASAGIDA kesinlesir, sonra yazilir)
                if cyc_pending is not None and not gec_bekliyor:
                    self._flush_cycle(cyc_pending, cyc_close_t, pend_info)
                    cyc_pending = None
                if cyc is not None and cyc["arm_dur"] is None:
                    cyc["arm_dur"] = cyc["dur"]   # buraya kadar = bekleme/gelis
                # CEVRIM ONCESI yapilan adimlari yeni cevrime tasi
                for _n, _t in list(pre_steps.items()):
                    if (now - _t) <= PRE_CARRY_SECS:
                        qa.mark(_n, "on")
                        print(f"[QA] '{pretty(_n)}' cevrim oncesinden tasindi "
                              f"({now - _t:.1f} sn once)")
                pre_steps.clear()
            elif motor_evt == "resume":
                # motor ayni yere geri dondu -> kapanma yanlisti, QA'yi KORU
                pass_since = None; closed_result = None; closed_missing = []
                gec_bekliyor = False
                print("[QA] motor ayni konuma dondu — kapanma geri alindi, "
                      "kontrol listesi korunuyor")
                if kep_yedek is not None:
                    kep_tracker.__dict__.update(kep_yedek[0])   # kep durumunu geri yukle
                    musur_gorulen = max(musur_gorulen, kep_yedek[1])
                    musur_poz = musur_poz or kep_yedek[2]
                    kep_yedek = None
                # cevrim sayaci da kaldigi yerden devam etsin (kayit yazilmaz)
                if cyc_pending is not None:
                    cyc = cyc_pending
                    cyc["last_t"] = now
                    cyc_pending = None
                elif cyc is None:
                    cyc = _yeni_cevrim(now)

            # ── GEC ADIM TOLERANSI: kapanistan sonra gelen eksik adimlar ──
            # Kontrol listesi yeni motor yerlesene kadar sifirlanmaz; model bu
            # arada eksik bir adimi gorurse (qa.done) biten cevrime yazilir.
            if gec_bekliyor:
                for _s in list(closed_missing):
                    if qa.done.get(_s, False):
                        closed_missing.remove(_s); gec_ok += 1
                        print(f"[QA] gec adim: '{pretty(_s)}' kapanistan "
                              f"{now - cyc_close_t:.1f} sn sonra goruldu -> biten cevrime yazildi")
                closed_result = "GECTI" if not closed_missing else "EKSIK"
                if cyc_pending is not None:
                    pend_info = (closed_result, gec_ok, gec_tot, list(closed_missing))
                if now >= gec_bitis or motor_evt == "arm":
                    gec_bekliyor = False
                    ok_c = gec_ok; tot_c = gec_tot
                    if closed_result == "GECTI":
                        print(f"[QA] MOTOR CIKTI -> CEVRIM KAPANDI: GECTI ({ok_c}/{tot_c})")
                    else:
                        eks = ", ".join(pretty(s) for s in closed_missing)
                        print(f"[QA] MOTOR CIKTI -> CEVRIM KAPANDI: EKSIK -> {eks}")
                        # eksik cevrimin son dakikalarini video olarak sakla
                        if KLIP_AKTIF and self.klip_ac.get():
                            yapilan = ", ".join(pretty(s) for s in QA_STEP_CLASSES
                                                if s not in closed_missing)
                            klip_cyc = cyc_pending if cyc_pending is not None else son_kapanan_cyc
                            kb, kbit, neden = eksik_penceresi(klip_cyc, closed_missing, cyc_close_t)
                            print(f"[KLIP] eksik penceresi: {kbit-kb:.1f} sn — {neden}")
                            self.klip_kaydet(
                                "eksik", ek=closed_missing[0] if closed_missing else "",
                                aralik=(kb, kbit),
                                bilgi={"baslik": "EKSIK: " + eks,
                                       "zaman": datetime.fromtimestamp(cyc_close_t)
                                                .strftime("%Y-%m-%d %H:%M:%S"),
                                       "eksik adimlar": eks,
                                       "yapilan adimlar": yapilan or "-",
                                       "tamamlanan": f"{ok_c}/{tot_c}",
                                       "klip araligi": (
                                           datetime.fromtimestamp(kb).strftime("%H:%M:%S")
                                           + " - "
                                           + datetime.fromtimestamp(kbit).strftime("%H:%M:%S")
                                           + f"  ({kbit-kb:.1f} sn)"),
                                       "pencere nasil secildi": neden,
                                       "cevrim suresi sn": (f"{klip_cyc['dur']:.1f}"
                                                            if klip_cyc else "?")})

            # bekleyen cevrim kaydi: geri alma penceresi gecti -> dosyaya yaz
            if (cyc_pending is not None and not gec_bekliyor
                    and (now - cyc_close_t) > max(MOTOR_UNDO_SECS, GEC_ADIM_SECS)):
                self._flush_cycle(cyc_pending, cyc_close_t, pend_info)
                cyc_pending = None

            qa_done, qa_src, qa_ok, qa_total, qa_passed = qa.snapshot()

            # sonuc gosterimi ve otomatik sifirlama
            reset_in = None
            if pass_since is not None:
                # gec adim penceresi bitmeden liste sifirlanmaz
                reset_in = max(0.0, max(AUTO_RESET_SECS, GEC_ADIM_SECS) - (now - pass_since))
                if reset_in <= 0:
                    _full_reset("sure doldu, yeni parca bekleniyor")
                    pass_since = None; reset_in = None
                    closed_result = None; closed_missing = []
                    qa_done, qa_src, qa_ok, qa_total, qa_passed = qa.snapshot()

            # cizim (bolgeler + motor gostergesi)
            with self.zone_lock:
                if last_annotated is not None:
                    if self.show_zones.get():
                        for z in self.zones:
                            last_annotated = draw_zone(last_annotated, z, fw, fh)
                        if self.drag_rect is not None:
                            x1 = int(self.drag_rect[0]*fw); y1 = int(self.drag_rect[1]*fh)
                            x2 = int(self.drag_rect[2]*fw); y2 = int(self.drag_rect[3]*fh)
                            cv2.rectangle(last_annotated, (x1,y1), (x2,y2), (255,255,255), 2, cv2.LINE_AA)
                    if self.show_motor_gate.get():
                        last_annotated = draw_motor_gate(last_annotated, motor_mon, move_thres, fw, fh)
                    # KEP SOKME: her zaman goster (slot sayaci + support ROI)
                    last_annotated = draw_kep_tracker(
                        last_annotated, kep_tracker,
                        support_rect(kep_tracker.motor_box), fw, fh,
                        musur_ok=bool(qa.done.get(MUSUR_TAKILI_ADIM, False)),
                        musur_poz=musur_poz)
                    self.last_annotated = last_annotated

            # teshis: saniyede bir durum yaz
            if now - dbg_t >= 1.0:
                dbg_t = now
                z0 = self.zones[0] if self.zones else None
                zinfo = ""
                if z0 is not None:
                    zinfo = (f"el_bolgede={z0.hand_inside} hrk={z0.motion_score:.1f}"
                             f"/{z0.motion_thres:.1f} tetik={z0.triggered} | ")
                print(f"[TESHIS] kisi={n_person} el={len(hand_points)} | "
                      f"{zinfo}motor={motor_mon.state} kutu={len(motor_cands)} "
                      f"kayma={motor_mon.displacement:.3f}/{move_thres:.2f} "
                      f"yok={motor_mon.lost_for:.1f}sn | "
                      f"qa={n_qa} adim={steps_seen} "
                      f"is={sorted(set(work_seen) - set(steps_seen))}")

            last_stats = dict(n=last_n, n_qa=n_qa,
                              fps=fps_smoothed, qa_done=qa_done, qa_src=qa_src,
                              qa_ok=qa_ok, qa_total=qa_total, qa_passed=qa_passed,
                              qa_prog=qa_prog, qa_wait=qa_wait,
                              reset_in=reset_in, result=closed_result,
                              missing=[pretty(s) for s in closed_missing],
                              zones=zone_state, hands=len(hand_points),
                              n_person=n_person,
                              motor_state=motor_mon.state,
                              motor_disp=motor_mon.displacement,
                              motor_thres=move_thres,
                              motor_present=motor_mon.present,
                              motor_lost_for=motor_mon.lost_for,
                              motor_n=len(motor_cands),
                              aktif_sn=(cyc["dur"] if cyc is not None else 0.0),
                              aktif_islem=(cyc["va"] if cyc is not None else 0.0),
                              # ── KEP SOKME 6x6 ──────────────────────────
                              kep_done=list(kep_tracker.done),
                              kep_tamamlanan=kep_tracker.tamamlanan,
                              kep_toplam=kep_tracker.n,
                              kep_metin=kep_tracker.durum_metni(),
                              kep_support=kep_tracker.support_n,
                              kep_motorda=kep_tracker.motor_n,
                              kep_kalan=(0 if kep_tracker.tamamlandi_mi
                                         else kep_tracker.n - min(kep_tracker.support_n, kep_tracker.n)),
                              kep_eksik_no=list(getattr(self, "son_eksik_kepler", [])),
                              musur_takili=bool(qa.done.get(MUSUR_TAKILI_ADIM, False)))
            # klip tamponu (KLIP_AKTIF ile acilir)
            if KLIP_AKTIF and not self.paused.get():
                self.clips.feed(now, last_annotated)

            self.root.after(0, self._update_display, last_annotated, last_stats)

    # ── display ─────────────────────────────────────────────────────────
    def _update_display(self, frame_bgr, st):
        T = TEMA
        # ── ust serit ──
        self.fps_text.set(f"{st['fps']:.1f} fps")
        self.det_text.set(f"Tespit  islem {st.get('n_qa',0)}"
                          f" · motor {st.get('motor_n',0)}")
        self.hands_text.set(f"Kisi {st.get('n_person',0)} · el noktasi {st.get('hands',0)}")
        self.canli_nokta.itemconfig(
            self._nokta, fill=T["yesil"] if st["fps"] > 0.5 else T["kirmizi"])
        if KLIP_AKTIF:
            self.klip_text.set(f"Klip tamponu: {self.clips.sure:.0f} sn · "
                               f"{self.clips.mb:.0f} MB")

        # ── aktif cevrim ──
        a_sn = st.get("aktif_sn", 0.0)
        if a_sn > 0:
            a_is = st.get("aktif_islem", 0.0)
            self.aktif_buyuk.config(text=f"{sg(a_sn)} {sb()}", fg=T["cyan"])
            self.aktif_alt.config(text=f"{nf(a_sn)} sn · islem "
                                       f"{sg(a_is, 1)} {sb()}")
        else:
            self.aktif_buyuk.config(text="—", fg=T["cok_soluk"])
            self.aktif_alt.config(text="motor bekleniyor")

        # ── motor ──
        m_state = st.get("motor_state", "-")
        renk = {MotorCycleMonitor.STATE_RUN: T["yesil"],
                MotorCycleMonitor.STATE_SETTLING: T["sari"],
                MotorCycleMonitor.STATE_HIDDEN: T["turuncu"],
                MotorCycleMonitor.STATE_OUT: T["mor"]}.get(m_state, T["cok_soluk"])
        self.motor_nokta.itemconfig(self._mnokta, fill=renk)
        self.motor_ad.config(text=m_state, fg=renk)
        yok = ""
        if not st.get("motor_present", False) and st.get("motor_lost_for", 0.0) > 0.2:
            yok = f" · gorunmuyor {st['motor_lost_for']:.1f} sn"
        esik = max(1e-6, st.get("motor_thres", 0.1))
        self.motor_text.set(f"kayma {st.get('motor_disp',0.0):.3f} / {esik:.2f}{yok}")
        self._bar_ciz(self.motor_bar, min(1.0, st.get("motor_disp", 0.0) / esik),
                      T["turuncu"] if st.get("motor_disp", 0) > esik * 0.7 else T["cyan"])

        # ── bolgeler ──
        for zad, (nk, lbl, ek) in self.zone_labels.items():
            ic, trig, prog = st["zones"].get(zad, (False, False, 0.0))
            if trig:
                nk.itemconfig(1, fill=T["yesil"]); lbl.config(fg=T["metin"])
                ek.config(text="KAPI ACIK", fg=T["yesil"])
            elif ic:
                nk.itemconfig(1, fill=T["sari"]); lbl.config(fg=T["metin"])
                ek.config(text=f"%{prog*100:.0f}", fg=T["sari"])
            else:
                nk.itemconfig(1, fill=T["cizgi"]); lbl.config(fg=T["cok_soluk"])
                ek.config(text="", fg=T["cok_soluk"])

        # ── QA adimlari ──
        qa_prog = st.get("qa_prog", {}); qa_wait = st.get("qa_wait", {})
        for step, (nk, lbl, ek) in self.qa_labels.items():
            done = st["qa_done"].get(step, False)
            src = st["qa_src"].get(step, "")
            yildiz = " *" if step in PRE_CYCLE_STEPS else ""
            if done:
                rozet = {"bolge": "bolge", "on": "motor oncesi"}.get(src, "model")
                nk.itemconfig(1, fill=T["yesil"])
                lbl.config(text="v " + pretty(step), fg=T["metin"])
                # kep_sokme icin 6/6 bilgisi goster
                if step == "kep_sokme":
                    kt = st.get("kep_tamamlanan", KEP_COUNT)
                    kn = st.get("kep_toplam", KEP_COUNT)
                    rozet = f"{kt}/{kn} kep"
                ek.config(text=rozet, fg=T["yesil"])
            elif step in qa_wait:
                nk.itemconfig(1, fill=T["cizgi"])
                lbl.config(text=pretty(step), fg=T["cok_soluk"])
                ek.config(text=qa_wait[step], fg=T["cok_soluk"])
            elif step == "kep_sokme" and st.get("kep_tamamlanan", 0) > 0:
                # kep_sokme kismen tamamlandi: ilerleme goster
                kt = st.get("kep_tamamlanan", 0)
                kn = st.get("kep_toplam", KEP_COUNT)
                nk.itemconfig(1, fill=T["sari"])
                lbl.config(text=pretty(step), fg=T["metin"])
                ek.config(text=f"{kt}/{kn}", fg=T["sari"])
            elif qa_prog.get(step, 0.0) > 0.0:
                nk.itemconfig(1, fill=T["sari"])
                lbl.config(text=pretty(step), fg=T["metin"])
                ek.config(text=f"%{qa_prog[step]*100:.0f}", fg=T["sari"])
            else:
                nk.itemconfig(1, fill=T["cizgi"])
                lbl.config(text=pretty(step) + yildiz, fg=T["soluk"])
                ek.config(text="", fg=T["cok_soluk"])

        # ── QA halkasi + durum ──
        res = st.get("result"); ri = st.get("reset_in")
        toplam = max(1, st["qa_total"]); oran = st["qa_ok"] / toplam
        if res == "GECTI":
            hrenk, drenk, dmetin = T["yesil"], T["yesil"], "GECTI"
            alt = f"sifirlaniyor {ri:.0f} sn"
        elif res == "EKSIK":
            hrenk, drenk, dmetin = T["turuncu"], T["turuncu"], "EKSIK"
            eks = ", ".join(st.get("missing", [])[:2])
            alt = eks + ("..." if len(st.get("missing", [])) > 2 else "")
        else:
            bekleyen = [pretty(x) for x in QA_STEP_CLASSES if not st["qa_done"].get(x)]
            hrenk = T["cyan"] if oran > 0 else T["cok_soluk"]
            drenk, dmetin = (T["sari"], "BEKLIYOR")
            if not bekleyen:
                drenk, dmetin, alt = T["yesil"], "TAMAM", "motor cikisi bekleniyor"
            elif m_state == MotorCycleMonitor.STATE_WAIT:
                alt = "motor bekleniyor"
            else:
                alt = ", ".join(bekleyen[:2]) + ("..." if len(bekleyen) > 2 else "")
        self._halka_ciz(oran, toplam, hrenk)
        self.qa_durum.config(text=dmetin, fg=drenk)
        self.qa_alt.config(text=alt)

        # ── AGIR KISIM: kareyi Tk'ye cizmek ──
        # Pencere simge durumundaysa, kullanici kapattiysa ya da hiz siniri
        # dolmadiysa cizme; panel sayilari yine de guncellenir.
        simdi = time.time()
        hiz = TK_VIDEO_FPS if self.tk_video.get() else 0
        ciz = (hiz > 0 and (simdi - self._tk_son_kare) >= 1.0 / hiz
               and self.root.state() != "iconic")
        if not ciz:
            self._web_besle(frame_bgr, st, m_state, dmetin, alt)
            if hiz == 0 and not getattr(self, "_bos_ekran", False):
                self._bos_ekran = True
                self.canvas.configure(
                    image="", text=f"SUNUCU MODU\n\nGoruntu bu pencerede kapali.\n"
                                   f"{getattr(self.web, 'adres', 'web paneli')}\n\n"
                                   "Ayarlar > Gorunum'den geri acabilirsin.",
                    fg=TEMA["cok_soluk"], font=("Segoe UI", 11), justify="center")
            return
        self._bos_ekran = False
        self._tk_son_kare = simdi

        cw = self.canvas.winfo_width() or 960; ch = self.canvas.winfo_height() or 540
        h, w = frame_bgr.shape[:2]; scale = min(cw/w, ch/h)
        nw, nh = max(1,int(w*scale)), max(1,int(h*scale))
        resized = cv2.resize(frame_bgr, (nw, nh))
        self.disp_geom = ((cw - nw)//2, (ch - nh)//2, scale, w, h)

        if res == "GECTI":
            qa_col = (0, 170, 51); qa_txt = f"GECTI ({st['reset_in']:.0f})"
        elif res == "EKSIK":
            qa_col = (0, 119, 221); qa_txt = f"EKSIK ({st['reset_in']:.0f})"
        else:
            qa_col = (0, 34, 204); qa_txt = "BEKLIYOR"
        ov = resized.copy()
        cv2.rectangle(ov, (nw-235, 8), (nw-8, 74), (0,0,0), -1)
        resized = cv2.addWeighted(ov, 0.55, resized, 0.45, 0)
        cv2.putText(resized, f"QA {qa_txt} {st['qa_ok']}/{st['qa_total']}",
                    (nw-225, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, qa_col, 2, cv2.LINE_AA)
        cv2.putText(resized, f"MOTOR: {m_state}", (nw-225, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if self.edit_roi.get():
            cv2.putText(resized, "ROI DUZENLEME: goruntude surukle", (12, nh-14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.canvas.configure(image=img); self.canvas.image = img

        self._web_besle(frame_bgr, st, m_state, dmetin, alt)

    def _web_besle(self, kare, st, m_state, dmetin, alt):
        """Web paneline kare + durum gonderir (hiz sinirli)."""
        if not self.web:
            return
        self.web.kare_ver(kare)          # izleyici yoksa kendisi cikiyor
        if not self.web.izlenen:         # paneli acan yoksa durum da hazirlanmaz
            return
        simdi = time.time()
        if (simdi - self._web_son_durum) < 1.0 / max(1, WEB_DURUM_HZ):
            return
        self._web_son_durum = simdi
        try:
            self.web.durum_ver(self._web_durum(st, m_state, dmetin, alt, None))
        except Exception:
            pass

    def on_close(self):
        self.alive = False; self.cmd_q.put(("shutdown",))
        self.worker.join(timeout=3.0)
        self.root.destroy()


def main():
    # Sadece rapor uret:  python webcam15.py --rapor [YYYY-MM-DD]
    if "--rapor" in sys.argv or "--rapor-ai" in sys.argv:
        ai = "--rapor-ai" in sys.argv
        i = sys.argv.index("--rapor-ai" if ai else "--rapor")
        gun = sys.argv[i + 1] if len(sys.argv) > i + 1 else None
        if gun and gun.startswith("-"):
            gun = None
        rapor_olustur(CycleLogger(), gun, sunucu=ai)
        if ai:
            print("[rapor] sunucu acik — kapatmak icin Ctrl+C")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        return
    root = tk.Tk()
    app = App(root)
    app._refresh_cycle_stats()
    root.mainloop()


if __name__ == "__main__":
    main()
