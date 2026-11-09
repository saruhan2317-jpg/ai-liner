"""
AI-LINER — Web Panel Koprusu
================================================================
Bu modul, mevcut Tkinter uygulamasinin (App) YANINDA calisir.
YOLO cikarimi, motor cevrim mantigi, QA kilitleri gibi hicbir
ise yaramaz -> yaramaz is mantigina DOKUNMAZ. Sadece worker
thread'in zaten urettigi (frame_bgr, st) cifti App._update_display
icine girer girmez bunu "musluk" gibi yakalar ve:

  - Son kareyi JPEG'e cevirip bir MJPEG akisi olarak (/video_feed)
  - Durum sozlugunu (st) JSON'a cevirip bir SSE akisi olarak (/events)

tarayiciya yayinlar. Ayrica Tkinter'daki kontrollerle ayni
tk.Variable nesnelerini (conf, esik, checkbox vb.) web tarafindan
guvenli sekilde (root.after uzerinden) degistirebilen birkac
REST ucu sunar.

Kurulum:  pip install flask
Kullanim: ai_liner1.py icine 3 satirlik entegrasyon eklendi,
          ayrica bakiniz: README_WEB.md
"""
import json
import os
import socket
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_static")

# Web tarafindan degistirilmesine izin verilen tk.Variable kontrolleri.
# isim -> python tipi (deger bu tipe cast edilir, sonra var.set() ile yazilir)
CONTROLLABLE_VARS = {
    "conf_goggle": float,
    "conf_qa": float,
    "conf_kam": float,
    "conf_musur": float,
    "conf_motor": float,
    "dwell_kam": float,
    "dwell_musur": float,
    "motor_move": float,
    "motor_hold": float,
    "motor_lost": float,
    "motor_absent_closes": bool,
    "imgsz_idx": int,
    "alpha": float,
    "frame_skip": int,
    "paused": bool,
    "show_pose": bool,
    "show_goggle": bool,
    "show_earplug": bool,
    "show_zones": bool,
    "show_motor_gate": bool,
    "yuz_bulanik": bool,
    "yuz_kabalik": int,
}


def _sanitize(obj):
    """numpy sayilari / tuple gibi JSON'a dogrudan uymayan tipleri temizler."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 4)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


class WebBridge:
    def __init__(self, app, meta, host="0.0.0.0", port=8080):
        """
        app  : mevcut Tkinter App ornegi (self.root, self.cmd_q, kontrol
               degiskenleri (tk.Var'lar), self.zones, self.request_source,
               self.qa_reset_evt buradan kullanilir).
        meta : dashboard'un ihtiyac duydugu statik bilgiler (dict). Bkz.
               ai_liner1.py icindeki WebBridge(...) cagrisi.
        """
        self.app_ref = app
        self.meta = meta
        self.host, self.port = host, port

        self._lock = threading.Lock()
        self._frame_jpeg = None
        self._state = {}
        self._frame_event = threading.Event()
        self._srv = None

        self.flask = Flask(__name__, static_folder=None)
        self.flask.url_map.strict_slashes = False
        self._register_routes()

    # ── App._update_display tarafindan HER KAREDE cagrilir ──────────────
    def publish(self, frame_bgr, st):
        try:
            ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with self._lock:
                    self._frame_jpeg = buf.tobytes()
        except Exception as e:
            print("[web] jpeg kodlama hatasi:", e)

        # NOT: publish() App._update_display icinden, yani Tkinter'in ana
        # (UI) thread'inden cagrilir (root.after uzerinden) — bu yuzden
        # asagidaki tk.Variable okumalari guvenlidir.
        merged = _sanitize(dict(st))
        merged["_ts"] = time.time()
        try:
            merged["source_label"] = self.app_ref.source_label.get()
            merged["ip_status"] = self.app_ref.ip_status.get()
        except Exception:
            pass

        with self._lock:
            self._state = merged
        self._frame_event.set()

    # ── yasam dongusu ────────────────────────────────────────────────────
    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        from werkzeug.serving import make_server
        try:
            self._srv = make_server(self.host, self.port, self.flask, threaded=True)
        except OSError as e:
            print(f"[web] sunucu baslatilamadi (port {self.port} mesgul olabilir): {e}")
            return
        print(f"[web] AI-LINER paneli hazir  ->  http://{self._local_ip()}:{self.port}")
        self._srv.serve_forever()

    def stop(self):
        if self._srv is not None:
            try:
                self._srv.shutdown()
            except Exception:
                pass

    @staticmethod
    def _local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    # ── HTTP rotalari ────────────────────────────────────────────────────
    def _register_routes(self):
        f = self.flask
        a = self.app_ref

        @f.route("/")
        def index():
            return send_from_directory(STATIC_DIR, "index.html")

        @f.route("/<path:fn>")
        def static_files(fn):
            return send_from_directory(STATIC_DIR, fn)

        @f.route("/video_feed")
        def video_feed():
            return Response(
                self._mjpeg_gen(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @f.route("/events")
        def events():
            return Response(
                self._sse_gen(),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        @f.route("/api/state")
        def api_state():
            with self._lock:
                return jsonify(self._state)

        @f.route("/api/meta")
        def api_meta():
            data = dict(self.meta)
            data["zones"] = [z.name for z in a.zones]
            data["controls"] = self._current_control_values()
            data["ip_url"] = a.ip_url.get()
            data["webcams"] = self._webcam_list()
            data["videos"] = self._video_list()
            return jsonify(data)

        @f.route("/api/source", methods=["POST"])
        def api_source():
            data = request.get_json(force=True) or {}
            kind = data.get("kind")
            label = data.get("label") or kind
            if kind == "cam":
                sel = ("cam", int(data.get("index", 0)))
            elif kind == "video":
                sel = ("video", data.get("path"))
            elif kind == "ip":
                url = (data.get("url") or "").strip() or a.ip_url.get()
                a.root.after(0, a.ip_url.set, url)
                sel = ("ip", url)
            else:
                return jsonify({"ok": False, "error": "bilinmeyen kaynak turu"}), 400
            a.request_source(sel, label)
            return jsonify({"ok": True})

        @f.route("/api/stop", methods=["POST"])
        def api_stop():
            a.cmd_q.put(("stop",))
            return jsonify({"ok": True})

        @f.route("/api/reset", methods=["POST"])
        def api_reset():
            a.qa_reset_evt.set()
            return jsonify({"ok": True})

        @f.route("/api/set", methods=["POST"])
        def api_set():
            data = request.get_json(force=True) or {}
            name, value = data.get("name"), data.get("value")
            typ = CONTROLLABLE_VARS.get(name)
            if typ is None or not hasattr(a, name):
                return jsonify({"ok": False, "error": f"gecersiz kontrol: {name}"}), 400
            try:
                casted = typ(value)
            except Exception:
                return jsonify({"ok": False, "error": "gecersiz deger"}), 400
            var = getattr(a, name)
            a.root.after(0, var.set, casted)
            return jsonify({"ok": True, "name": name, "value": casted})

    def _webcam_list(self, max_index=4):
        """Hizli bir tarama ile bagli webcam indekslerini bulur.
        Not: video akisi acikken bu taramayi yapmak cihazi kilitleyebilir,
        bu yuzden sadece /api/meta ilk yuklemede cagrilir."""
        found = []
        for i in range(max_index):
            try:
                cap = cv2.VideoCapture(i)
                if cap is not None and cap.isOpened():
                    found.append(i)
                if cap is not None:
                    cap.release()
            except Exception:
                pass
        return found

    def _video_list(self):
        video_dir = self.meta.get("video_dir")
        if not video_dir or not os.path.isdir(video_dir):
            return []
        exts = (".mp4", ".avi", ".mkv", ".mov")
        return sorted(fn for fn in os.listdir(video_dir) if fn.lower().endswith(exts))

    def _current_control_values(self):
        a = self.app_ref
        out = {}
        for name in CONTROLLABLE_VARS:
            try:
                out[name] = _sanitize(getattr(a, name).get())
            except Exception:
                pass
        return out

    # ── akislar ──────────────────────────────────────────────────────────
    def _mjpeg_gen(self):
        boundary = b"--frame\r\n"
        while True:
            got = self._frame_event.wait(timeout=1.0)
            with self._lock:
                frame = self._frame_jpeg
            if got:
                self._frame_event.clear()
            if frame is None:
                time.sleep(0.05)
                continue
            yield (boundary + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    def _sse_gen(self):
        last_ts = 0
        while True:
            with self._lock:
                state = dict(self._state)
            ts = state.get("_ts", 0)
            if ts != last_ts:
                last_ts = ts
                yield f"data: {json.dumps(state)}\n\n"
            else:
                yield ": ping\n\n"
            time.sleep(0.08)
