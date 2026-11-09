(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const els = {
    urunAdi: $("urunAdi"), urunAlt: $("urunAlt"),
    cycleTime: $("cycleTime"), cycleSub: $("cycleSub"),
    fpsVal: $("fpsVal"), personVal: $("personVal"), connDot: $("connDot"),
    sourceLabel: $("sourceLabel"), webcamBtns: $("webcamBtns"), videoBtns: $("videoBtns"),
    ipUrl: $("ipUrl"), ipConnectBtn: $("ipConnectBtn"), ipStatus: $("ipStatus"),
    stopBtn: $("stopBtn"), resetBtn: $("resetBtn"),
    motorDot: $("motorDot"), motorState: $("motorState"), motorBar: $("motorBar"), motorCaption: $("motorCaption"),
    kepBar: $("kepBar"), kepPct: $("kepPct"),
    kepMotorBar: $("kepMotorBar"), kepMotorPct: $("kepMotorPct"),
    musurBar: $("musurBar"), musurPct: $("musurPct"), kepCaption: $("kepCaption"),
    settingsList: $("settingsList"),
    detChip: $("detChip"), resultBanner: $("resultBanner"), resultText: $("resultText"),
    ringFg: $("ringFg"), ringFrac: $("ringFrac"), ringState: $("ringState"), ringNote: $("ringNote"),
    qaList: $("qaList"), zoneList: $("zoneList"),
  };

  const RING_CIRC = 2 * Math.PI * 60; // r=60

  let META = null;
  let qaRowIndex = {};   // step -> {row, dotEl, badgeEl, nameEl}
  let zoneRowIndex = {}; // name -> {row, dotEl, badgeEl}
  let prevQaDone = {};

  const MOTOR_COLORS = {
    "MOTOR YOK": "var(--faint)",
    "YERLESIYOR": "var(--amber)",
    "POSTADA": "var(--green)",
    "GORUNMUYOR": "var(--orange)",
    "CIKTI": "var(--violet)",
  };

  function fmtSec(s, digits = 1) {
    if (s === undefined || s === null) return "—";
    return s.toFixed(digits);
  }

  // ── ayar tanimlari: /api/set uzerinden yazilan kontroller ──────────
  const SETTING_DEFS = [
    { name: "conf_qa", label: "İşlem eşiği", min: 0.05, max: 0.95, step: 0.01, kind: "range" },
    { name: "conf_motor", label: "Motor eşiği", min: 0.05, max: 0.95, step: 0.01, kind: "range" },
    { name: "dwell_kam", label: "Kam mili onay (sn)", min: 0, max: 3, step: 0.1, kind: "range" },
    { name: "motor_move", label: "Motor çıkış eşiği", min: 0.02, max: 0.5, step: 0.005, kind: "range" },
    { name: "motor_hold", label: "Kaybolma toleransı (sn)", min: 1, max: 30, step: 0.5, kind: "range" },
    { name: "alpha", label: "Kutu/maske opaklığı", min: 0, max: 1, step: 0.05, kind: "range" },
    { name: "frame_skip", label: "Kare atlama", min: 0, max: 6, step: 1, kind: "range" },
    { name: "show_pose", label: "İskelet çizimi", kind: "toggle" },
    { name: "show_zones", label: "Bölgeleri göster", kind: "toggle" },
    { name: "paused", label: "Duraklat", kind: "toggle" },
  ];

  function pretty(name) {
    if (META && META.pretty && META.pretty[name]) return META.pretty[name];
    return name;
  }

  // ── kurulum ─────────────────────────────────────────────────────────
  async function boot() {
    try {
      const res = await fetch("/api/meta");
      META = await res.json();
    } catch (e) {
      META = { qa_steps: [], zones: [], webcams: [], videos: [], controls: {} };
    }

    els.urunAdi.textContent = META.urun || "AI-LINER";
    els.urunAlt.textContent = META.alt || "";
    els.ipUrl.value = META.ip_url || "";

    buildSourceButtons();
    buildSettings();
    buildQaList();
    buildZoneList();
    wireGlobalButtons();
    connectEvents();
  }

  function buildSourceButtons() {
    els.webcamBtns.innerHTML = "";
    (META.webcams && META.webcams.length ? META.webcams : [0]).forEach((idx) => {
      const b = document.createElement("button");
      b.className = "btn";
      b.textContent = `Webcam ${idx}`;
      b.onclick = () => postJSON("/api/source", { kind: "cam", index: idx, label: `Webcam ${idx}` });
      els.webcamBtns.appendChild(b);
    });

    els.videoBtns.innerHTML = "";
    (META.videos || []).forEach((name) => {
      const b = document.createElement("button");
      b.className = "btn";
      b.textContent = name;
      b.onclick = () => postJSON("/api/source", { kind: "video", path: joinPath(META.video_dir, name), label: name });
      els.videoBtns.appendChild(b);
    });
  }

  function joinPath(dir, name) {
    if (!dir) return name;
    return dir.endsWith("/") || dir.endsWith("\\") ? dir + name : dir + "/" + name;
  }

  function buildSettings() {
    els.settingsList.innerHTML = "";
    const controls = META.controls || {};
    SETTING_DEFS.forEach((def) => {
      if (!(def.name in controls)) return;
      const row = document.createElement("div");
      row.className = "setting-row";

      if (def.kind === "range") {
        const val = Number(controls[def.name]);
        row.innerHTML = `
          <div class="setting-head"><span>${def.label}</span><span class="val" id="val_${def.name}">${val}</span></div>
          <input type="range" id="in_${def.name}" min="${def.min}" max="${def.max}" step="${def.step}" value="${val}">
        `;
        els.settingsList.appendChild(row);
        const input = row.querySelector(`#in_${def.name}`);
        const valEl = row.querySelector(`#val_${def.name}`);
        let t = null;
        input.addEventListener("input", () => {
          valEl.textContent = input.value;
          clearTimeout(t);
          t = setTimeout(() => postJSON("/api/set", { name: def.name, value: input.value }), 120);
        });
      } else {
        const checked = !!controls[def.name];
        row.innerHTML = `
          <div class="toggle-row">
            <span class="label">${def.label}</span>
            <label class="switch">
              <input type="checkbox" id="in_${def.name}" ${checked ? "checked" : ""}>
              <span class="track"></span>
            </label>
          </div>
        `;
        els.settingsList.appendChild(row);
        const input = row.querySelector(`#in_${def.name}`);
        input.addEventListener("change", () => {
          postJSON("/api/set", { name: def.name, value: input.checked });
        });
      }
    });
  }

  function buildQaList() {
    els.qaList.innerHTML = "";
    qaRowIndex = {};
    (META.qa_steps || []).forEach((step) => {
      const row = document.createElement("div");
      row.className = "qa-row wait";
      const star = (META.pre_cycle_steps || []).includes(step) ? " *" : "";
      row.innerHTML = `
        <span class="dot"></span>
        <span class="name">${pretty(step)}${star}</span>
        <span class="badge">bekliyor</span>
      `;
      els.qaList.appendChild(row);
      qaRowIndex[step] = {
        row,
        dot: row.querySelector(".dot"),
        name: row.querySelector(".name"),
        badge: row.querySelector(".badge"),
      };
      prevQaDone[step] = false;
    });
  }

  function buildZoneList() {
    els.zoneList.innerHTML = "";
    zoneRowIndex = {};
    const zones = META.zones || [];
    if (zones.length === 0) {
      els.zoneList.innerHTML = `<div class="ip-status">tanımlı bölge yok</div>`;
      return;
    }
    zones.forEach((name) => {
      const row = document.createElement("div");
      row.className = "zone-row";
      row.innerHTML = `<span class="dot"></span><span class="name">${name}</span><span class="badge"></span>`;
      els.zoneList.appendChild(row);
      zoneRowIndex[name] = { row, dot: row.querySelector(".dot"), badge: row.querySelector(".badge") };
    });
  }

  function wireGlobalButtons() {
    els.ipConnectBtn.onclick = () => {
      const url = els.ipUrl.value.trim();
      els.ipStatus.textContent = "Bağlanılıyor...";
      postJSON("/api/source", { kind: "ip", url, label: url });
    };
    els.stopBtn.onclick = () => postJSON("/api/stop", {});
    els.resetBtn.onclick = () => postJSON("/api/reset", {});
  }

  async function postJSON(url, body) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      return await res.json();
    } catch (e) {
      console.error("istek hatasi:", url, e);
    }
  }

  // ── canli durum akisi ─────────────────────────────────────────────

  function connectEvents() {
    const src = new EventSource("/events");
    let watchdog = setTimeout(markOffline, 3000);

    src.onmessage = (ev) => {
      clearTimeout(watchdog);
      watchdog = setTimeout(markOffline, 3000);
      markOnline();
      try {
        applyState(JSON.parse(ev.data));
      } catch (e) { /* ping veya bozuk kare — yoksay */ }
    };
    src.onerror = () => { markOffline(); };
  }

  function markOnline() { els.connDot.classList.add("live"); }
  function markOffline() { els.connDot.classList.remove("live"); }

  function applyState(st) {
    if (!st || !st._ts) return;

    els.fpsVal.textContent = fmtSec(st.fps, 1);
    els.personVal.textContent = st.n_person ?? 0;
    const kepTop = st.kep_toplam ?? 6;
    els.detChip.textContent =
      `qa ${st.n_qa ?? 0} · motor ${st.motor_n ?? 0} · kep ${Math.min(st.kep_support ?? 0, kepTop)}/${kepTop}`;

    // ── aktif çevrim süresi ──
    if (st.aktif_sn && st.aktif_sn > 0) {
      els.cycleTime.textContent = `${fmtSec(st.aktif_sn)} sn`;
      els.cycleSub.textContent = `işlem ${fmtSec(st.aktif_va ?? st.aktif_islem)} sn`;
    } else {
      els.cycleTime.textContent = "—";
      els.cycleSub.textContent = "motor bekleniyor";
    }

    // ── motor ──
    const mState = st.motor_state || "-";
    const mColor = MOTOR_COLORS[mState] || "var(--faint)";
    els.motorDot.style.background = mColor;
    els.motorState.textContent = mState;
    els.motorState.style.color = mColor;
    const thres = Math.max(1e-6, st.motor_thres ?? 0.1);
    const disp = st.motor_disp ?? 0;
    const ratio = Math.min(1, disp / thres);
    els.motorBar.style.width = `${(ratio * 100).toFixed(1)}%`;
    els.motorBar.style.background = ratio > 0.7 ? "var(--orange)" : "var(--cyan)";
    let cap = `kayma ${disp.toFixed(3)} / ${thres.toFixed(2)}`;
    if (!st.motor_present && (st.motor_lost_for ?? 0) > 0.2) {
      cap += ` · görünmüyor ${fmtSec(st.motor_lost_for)} sn`;
    }
    els.motorCaption.textContent = cap;

    // ── Kep & Müşür (v8) ──
    setKep(st);

    // ── QA satirlari ──
    const qaDone = st.qa_done || {};
    const qaSrc = st.qa_src || {};
    const qaProg = st.qa_prog || {};
    const qaWait = st.qa_wait || {};
    Object.keys(qaRowIndex).forEach((step) => {
      const ui = qaRowIndex[step];
      const done = !!qaDone[step];
      ui.row.classList.remove("done", "progress", "wait");
      if (done) {
        ui.row.classList.add("done");
        const src = qaSrc[step];
        ui.badge.textContent = src === "bolge" ? "bölge" : src === "on" ? "motor öncesi" : "model";
        if (!prevQaDone[step]) {
          ui.row.classList.add("flash");
          setTimeout(() => ui.row.classList.remove("flash"), 900);
        }
      } else if (qaWait[step]) {
        ui.row.classList.add("wait");
        ui.badge.textContent = qaWait[step];
      } else if ((qaProg[step] || 0) > 0) {
        ui.row.classList.add("progress");
        ui.badge.textContent = `%${Math.round(qaProg[step] * 100)}`;
      } else {
        ui.row.classList.add("wait");
        ui.badge.textContent = "bekliyor";
      }
      prevQaDone[step] = done;
    });

    // ── bolgeler ──
    const zones = st.zones || {};
    Object.keys(zoneRowIndex).forEach((name) => {
      const ui = zoneRowIndex[name];
      const z = zones[name];
      ui.row.classList.remove("open", "active");
      if (!z) { ui.badge.textContent = ""; return; }
      const [inside, triggered, progress] = z;
      if (triggered) {
        ui.row.classList.add("open");
        ui.badge.textContent = "KAPI AÇIK";
      } else if (inside) {
        ui.row.classList.add("active");
        ui.badge.textContent = `%${Math.round((progress || 0) * 100)}`;
      } else {
        ui.badge.textContent = "";
      }
    });

    // ── sonuç halkası + banner ──
    const total = Math.max(1, st.qa_total ?? 1);
    const ok = st.qa_ok ?? 0;
    const ratioRing = ok / total;
    els.ringFrac.textContent = `${ok}/${total}`;
    const dashoffset = RING_CIRC * (1 - ratioRing);
    els.ringFg.style.strokeDashoffset = String(dashoffset);

    const result = st.result;
    els.resultBanner.classList.remove("ok", "eksik", "wait");
    if (result === "GECTI") {
      els.ringFg.style.stroke = "var(--green)";
      els.ringState.textContent = `GEÇTİ · ${fmtSec(st.reset_in, 0)} sn`;
      els.ringNote.textContent = "sıfırlanıyor";
      els.resultText.textContent = `GEÇTİ (${fmtSec(st.reset_in, 0)})`;
      els.resultBanner.classList.add("ok");
    } else if (result === "EKSIK") {
      els.ringFg.style.stroke = "var(--orange)";
      els.ringState.textContent = "EKSİK";
      const missing = (st.missing || []).map(pretty);
      els.ringNote.textContent = missing.slice(0, 2).join(", ") + (missing.length > 2 ? "..." : "");
      els.resultText.textContent = `EKSİK (${fmtSec(st.reset_in, 0)})`;
      els.resultBanner.classList.add("eksik");
    } else {
      els.ringFg.style.stroke = "var(--cyan)";
      els.ringState.textContent = ratioRing > 0 ? "DEVAM EDİYOR" : "BEKLİYOR";
      els.ringNote.textContent = st.motor_state === "MOTOR YOK" ? "motor bekleniyor" : "";
      els.resultText.textContent = "BEKLİYOR";
      els.resultBanner.classList.add("wait");
    }

    if (st.source_label) els.sourceLabel.textContent = st.source_label;
    if (st.ip_status) els.ipStatus.textContent = st.ip_status;
  }

  function setKep(st) {
    if (!els.kepBar) return;
    const top = Math.max(1, st.kep_toplam ?? 6);
    const sup = Math.min(st.kep_support ?? 0, top);
    const mot = Math.min(st.kep_kalan ?? Math.max(0, top - sup), top);   // motorda kalan kep
    const tamam = sup >= top || !!(st.qa_done || {}).kep_sokme;
    els.kepBar.style.width = `${(sup / top * 100).toFixed(0)}%`;
    els.kepBar.style.background = tamam ? "var(--green)" : "var(--cyan)";
    els.kepPct.textContent = `${sup}/${top}`;
    els.kepPct.style.color = tamam ? "var(--green)" : "var(--text)";
    els.kepMotorBar.style.width = `${(mot / top * 100).toFixed(0)}%`;
    els.kepMotorBar.style.background = mot > 0 ? "var(--amber)" : "var(--faint)";
    els.kepMotorPct.textContent = String(mot);
    const mus = !!st.musur_takili;
    els.musurBar.style.width = mus ? "100%" : "0%";
    els.musurBar.style.background = "var(--green)";
    els.musurPct.textContent = mus ? "TAKILI" : "—";
    els.musurPct.style.color = mus ? "var(--green)" : "var(--faint)";
    const eks = st.kep_eksik_no || [];
    els.kepCaption.textContent = tamam ? "kep sökme tamam"
      : eks.length ? `son motorda kalan kep: ${eks.join(", ")}`
      : (st.motor_state === "POSTADA" ? "kep sökülüyor" : "kep sökme bekleniyor");
  }

  function setPpe(barEl, pctEl, ratio, green) {
    const r = ratio || 0;
    barEl.style.width = `${Math.min(100, r * 100).toFixed(0)}%`;
    barEl.style.background = green ? "var(--green)" : "var(--red)";
    pctEl.textContent = `%${Math.round(r * 100)}`;
    pctEl.style.color = green ? "var(--green)" : "var(--red)";
  }

  boot();
})();
