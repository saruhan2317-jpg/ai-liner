"""AI-LINER yama uygulayici (Windows'ta 'patch' komutu olmadigi icin).

Kullanim (gozlukv klasorunde):
    python yama_uygula.py ai_liner6_to_8.patch     -> ai_liner8.py olusturur (ai_liner6.py DEGISMEZ)
    python yama_uygula.py web_panel_v8.patch       -> web_static dosyalarini gunceller (.bak yedegi alir)

Yama, dosyadaki satirlarla birebir eslesmezse HICBIR dosyaya yazmaz ve hata verir.
"""
import os, re, shutil, sys

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def dosyalari_oku(yama):
    satirlar = open(yama, encoding="utf-8").read().split("\n")
    parcalar, i = [], 0
    while i < len(satirlar):
        if satirlar[i].startswith("--- ") and i + 1 < len(satirlar) and satirlar[i + 1].startswith("+++ "):
            eski = satirlar[i][4:].split("\t")[0].strip()
            yeni = satirlar[i + 1][4:].split("\t")[0].strip()
            i += 2; hunks = []
            while i < len(satirlar) and not satirlar[i].startswith("--- "):
                m = HUNK.match(satirlar[i])
                if m:
                    bas = int(m.group(1)); i += 1; govde = []
                    while i < len(satirlar) and satirlar[i][:1] in (" ", "-", "+", "\\") and not HUNK.match(satirlar[i]):
                        if satirlar[i].startswith("--- ") and i + 1 < len(satirlar) and satirlar[i + 1].startswith("+++ "):
                            break
                        govde.append(satirlar[i]); i += 1
                    hunks.append((bas, govde))
                else:
                    i += 1
            parcalar.append((eski, yeni, hunks))
        else:
            i += 1
    return parcalar


def yol(ad):
    ad = ad.replace("\\", "/")
    if ad.startswith(("a/", "b/")):
        ad = ad[2:]
    return os.path.normpath(ad)


def uygula(eski_metin, hunks, ad):
    eski = eski_metin.split("\n")
    cikti, poz = [], 0
    for bas, govde in hunks:
        hedef = max(0, bas - 1)
        cikti.extend(eski[poz:hedef]); poz = hedef
        for s in govde:
            if s.startswith("\\"):
                continue
            isaret, metin = s[:1], s[1:]
            if isaret in (" ", "-"):
                if poz >= len(eski) or eski[poz].rstrip("\r") != metin:
                    raise SystemExit(f"HATA: {ad} satir {poz + 1} yamayla uyusmuyor. Hicbir dosya degistirilmedi.")
                if isaret == " ":
                    cikti.append(eski[poz])
                poz += 1
            elif isaret == "+":
                cikti.append(metin)
    cikti.extend(eski[poz:])
    return "\n".join(cikti)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    yazilacak = []
    for eski_ad, yeni_ad, hunks in dosyalari_oku(sys.argv[1]):
        eski_yol, yeni_yol = yol(eski_ad), yol(yeni_ad)
        eski_metin = open(eski_yol, encoding="utf-8", newline="").read() if os.path.exists(eski_yol) else ""
        crlf = "\r\n" in eski_metin
        yeni_metin = uygula(eski_metin.replace("\r\n", "\n"), hunks, eski_yol)
        if crlf:
            yeni_metin = yeni_metin.replace("\n", "\r\n")
        yazilacak.append((eski_yol, yeni_yol, yeni_metin))
    for eski_yol, yeni_yol, metin in yazilacak:          # once hepsi dogrulandi, simdi yaz
        if os.path.exists(yeni_yol):
            shutil.copy2(yeni_yol, yeni_yol + ".bak")
        os.makedirs(os.path.dirname(yeni_yol) or ".", exist_ok=True)
        open(yeni_yol, "w", encoding="utf-8", newline="").write(metin)
        print(f"yazildi: {yeni_yol}" + ("  (eski hali .bak)" if os.path.exists(yeni_yol + ".bak") else ""))


if __name__ == "__main__":
    main()
