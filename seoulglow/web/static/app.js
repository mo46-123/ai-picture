/* SeoulGlow front end — plain JS, no build step. */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const SLIDERS = [
    ["smooth", "磨皮"], ["whiten", "美白"], ["slim", "瘦脸"], ["eyes", "大眼"],
    ["nose", "瘦鼻"], ["makeup", "妆容"], ["filter", "滤镜"],
  ];
  const PREVIEW_SIDE = 1280;   // what we send while you adjust (fast)
  const FULL_SIDE = 2048;      // what we send when you save / use AI

  const state = {
    presets: [], shades: [], style: null, lip: null, values: {}, intensity: 100,
    preview: null, full: null, originalURL: null, resultURL: null,
    mode: "local", aiReady: false, controller: null, timer: null,
    pendingGlow: false, maxUploadMb: 15,
  };

  const el = {
    stage: $("#stage"), pick: $("#pick"), file: $("#file"), photo: $("#photo"),
    result: $("#result"), original: $("#original"), badge: $("#badge"), compare: $("#compare"),
    busy: $("#busy"), busyText: $("#busyText"), notice: $("#notice"),
    pans: $("#pans"), styleDesc: $("#styleDesc"), tints: $("#tints"), lipName: $("#lipName"),
    intensity: $("#intensity"), intensityOut: $("#intensityOut"), fine: $("#fineSliders"),
    again: $("#again"), ai: $("#aiBtn"), save: $("#save"), aiNote: $("#aiNote"),
    sheet: $("#sheet"), saveImg: $("#saveImg"), download: $("#download"), closeSheet: $("#closeSheet"),
    toast: $("#toast"),
  };

  // ---------------------------------------------------------------- helpers
  let toastTimer = null;
  function toast(message, ms = 3200) {
    el.toast.textContent = message;
    el.toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.toast.hidden = true; }, ms);
  }

  function setNotice(text) { el.notice.textContent = text || ""; }

  function setBusy(on, text = "正在修图…") {
    el.busyText.textContent = text;
    el.busy.hidden = !on;
    el.stage.setAttribute("aria-busy", on ? "true" : "false");
  }

  function paintRange(input) {
    const p = ((input.value - input.min) / (input.max - input.min)) * 100;
    input.style.setProperty("--p", `${p}%`);
  }

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });
  }

  // Downscale + re-encode in the browser: smaller uploads, EXIF rotation baked in,
  // and HEIC photos become JPEG on phones that can decode them.
  function toJpeg(img, maxSide, quality) {
    const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
    const w = Math.max(1, Math.round(img.naturalWidth * scale));
    const h = Math.max(1, Math.round(img.naturalHeight * scale));
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, w, h);
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("encode failed"))), "image/jpeg", quality);
    });
  }

  async function postForm(url, fields, signal) {
    const form = new FormData();
    for (const [key, value] of Object.entries(fields)) {
      if (value instanceof Blob) form.append(key, value, "photo.jpg");
      else if (value !== null && value !== undefined) form.append(key, String(value));
    }
    const resp = await fetch(url, { method: "POST", body: form, signal });
    let data = {};
    try { data = await resp.json(); } catch (_) { /* non-JSON error page */ }
    if (!resp.ok) throw new Error(data.error || "出了点问题，请稍后再试");
    return data;
  }

  function currentFields(image, preview) {
    const fields = { image, style: state.style, lip: state.lip || "none",
                     intensity: (state.intensity / 100).toFixed(2), preview: preview ? 1 : 0 };
    for (const [key] of SLIDERS) fields[key] = (state.values[key] / 100).toFixed(2);
    return fields;
  }

  // ----------------------------------------------------------- build the UI
  function preset(id) { return state.presets.find((p) => p.id === id); }

  function buildPans() {
    el.pans.innerHTML = "";
    for (const p of state.presets) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pan";
      btn.setAttribute("role", "radio");
      btn.dataset.id = p.id;
      btn.innerHTML = `<span class="pan-disc" aria-hidden="true"></span><span class="pan-name"></span>`;
      btn.querySelector(".pan-disc").style.cssText = `--a:${p.swatch[0]};--b:${p.swatch[1]}`;
      btn.querySelector(".pan-name").textContent = p.name;
      btn.addEventListener("click", () => selectStyle(p.id, true));
      el.pans.appendChild(btn);
    }
  }

  function buildTints() {
    el.tints.innerHTML = "";
    const options = [...state.shades, { id: "none", name: "不涂口红", hex: null }];
    for (const s of options) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = s.hex ? "tint" : "tint none";
      btn.setAttribute("role", "radio");
      btn.setAttribute("aria-label", s.name);
      btn.title = s.name;
      btn.dataset.id = s.id;
      if (s.hex) btn.style.setProperty("--c", s.hex);
      btn.addEventListener("click", () => { selectLip(s.id); schedule(120); });
      el.tints.appendChild(btn);
    }
  }

  function buildSliders() {
    el.fine.innerHTML = "";
    for (const [key, label] of SLIDERS) {
      const row = document.createElement("div");
      row.className = "slider";
      row.innerHTML = `<label for="s-${key}">${label}</label>
        <input type="range" id="s-${key}" min="0" max="100" value="50">
        <output for="s-${key}">50</output>`;
      const input = row.querySelector("input");
      const out = row.querySelector("output");
      input.addEventListener("input", () => {
        state.values[key] = Number(input.value);
        out.textContent = input.value;
        paintRange(input);
        schedule(280);
      });
      input.addEventListener("change", () => schedule(0));
      el.fine.appendChild(row);
    }
    el.intensity.addEventListener("input", () => {
      state.intensity = Number(el.intensity.value);
      el.intensityOut.textContent = el.intensity.value;
      paintRange(el.intensity);
      schedule(280);
    });
    el.intensity.addEventListener("change", () => schedule(0));
    paintRange(el.intensity);
  }

  function selectLip(id) {
    state.lip = id;
    for (const btn of el.tints.children) btn.setAttribute("aria-checked", String(btn.dataset.id === id));
    const shade = state.shades.find((s) => s.id === id);
    el.lipName.textContent = shade ? shade.name : "不涂";
  }

  function selectStyle(id, render) {
    const p = preset(id);
    if (!p) return;
    state.style = id;
    for (const btn of el.pans.children) btn.setAttribute("aria-checked", String(btn.dataset.id === id));
    el.styleDesc.textContent = `${p.tagline}。${p.description}`;
    for (const [key] of SLIDERS) {
      const v = Math.round(p.defaults[key] * 100);
      state.values[key] = v;
      const input = $(`#s-${key}`);
      input.value = v;
      input.nextElementSibling.textContent = v;
      paintRange(input);
    }
    selectLip(p.defaults.lip);
    if (render && state.preview) {
      state.pendingGlow = true;
      schedule(0);
    }
  }

  // ------------------------------------------------------------ rendering
  function schedule(delay) {
    if (!state.preview) return;
    clearTimeout(state.timer);
    state.timer = setTimeout(renderLocal, delay);
  }

  function showResult(url, mode) {
    state.resultURL = url;
    state.mode = mode;
    el.result.src = url;
    el.badge.hidden = mode !== "ai";
    el.save.disabled = false;
    if (state.pendingGlow && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      el.photo.classList.remove("glow");
      void el.photo.offsetWidth; // restart the animation
      el.photo.classList.add("glow");
    }
    state.pendingGlow = false;
  }

  async function renderLocal() {
    if (!state.preview) return;
    if (state.controller) state.controller.abort();
    const controller = new AbortController();
    state.controller = controller;
    setBusy(true);
    try {
      const data = await postForm("/api/retouch", currentFields(state.preview, true), controller.signal);
      if (controller.signal.aborted) return;
      showResult(data.image, "local");
      setNotice(data.notice || (data.faces > 1 ? `已为 ${data.faces} 张脸修图` : ""));
    } catch (err) {
      if (err.name === "AbortError") return;
      toast(err.message === "Failed to fetch" ? "网络好像断了，检查一下再试" : err.message);
    } finally {
      if (state.controller === controller) {
        state.controller = null;
        setBusy(false);
      }
    }
  }

  async function openFile(file) {
    if (!file) return;
    if (file.size > state.maxUploadMb * 1024 * 1024 * 3) {
      toast("这张照片太大了，换一张试试");
      return;
    }
    const url = URL.createObjectURL(file);
    let img;
    try {
      img = await loadImage(url);
    } catch (_) {
      URL.revokeObjectURL(url);
      toast("这张图片打不开，换一张 JPG 或 PNG 试试");
      return;
    }
    try {
      state.full = await toJpeg(img, FULL_SIDE, 0.92);
      state.preview = await toJpeg(img, PREVIEW_SIDE, 0.9);
    } finally {
      URL.revokeObjectURL(url);
    }
    if (state.originalURL) URL.revokeObjectURL(state.originalURL);
    state.originalURL = URL.createObjectURL(state.preview);
    el.original.src = state.originalURL;
    el.result.src = state.originalURL;
    el.badge.hidden = true;
    el.pick.hidden = true;
    el.photo.hidden = false;
    setNotice("");
    state.pendingGlow = true;
    schedule(0);
  }

  // --------------------------------------------------------- compare (hold)
  function holdOriginal(on) {
    if (!state.originalURL) return;
    el.original.hidden = !on;
    el.compare.classList.toggle("holding", on);
    el.compare.textContent = on ? "松开看效果" : "按住看原图";
  }

  el.compare.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    el.compare.setPointerCapture(e.pointerId);
    holdOriginal(true);
  });
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
    el.compare.addEventListener(type, () => holdOriginal(false));
  }
  el.compare.addEventListener("keydown", (e) => {
    if ((e.key === " " || e.key === "Enter") && !e.repeat) { e.preventDefault(); holdOriginal(true); }
  });
  el.compare.addEventListener("keyup", () => holdOriginal(false));
  el.compare.addEventListener("contextmenu", (e) => e.preventDefault());

  // ------------------------------------------------------------- actions
  el.pick.addEventListener("click", () => el.file.click());
  el.again.addEventListener("click", () => el.file.click());
  el.file.addEventListener("change", () => {
    openFile(el.file.files[0]);
    el.file.value = "";
  });

  // Desktop drag & drop
  for (const type of ["dragenter", "dragover"]) {
    el.stage.addEventListener(type, (e) => { e.preventDefault(); el.stage.classList.add("dragging"); });
  }
  for (const type of ["dragleave", "drop"]) {
    el.stage.addEventListener(type, () => el.stage.classList.remove("dragging"));
  }
  el.stage.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer && e.dataTransfer.files[0];
    if (file) openFile(file);
  });

  el.ai.addEventListener("click", async () => {
    if (!state.full) { toast("先选一张自拍"); return; }
    if (!state.aiReady) {
      toast("AI 写真还没开通（需要在服务器上配置大模型 API Key），先用本地精修吧", 4200);
      return;
    }
    if (state.controller) state.controller.abort();
    clearTimeout(state.timer);
    const controller = new AbortController();
    state.controller = controller;
    setBusy(true, "AI 写真生成中，通常要 10–40 秒…");
    el.ai.disabled = true;
    try {
      const data = await postForm("/api/ai", { image: state.full, style: state.style }, controller.signal);
      state.pendingGlow = true;
      showResult(data.image, data.fallback ? "local" : "ai");
      setNotice(data.notice || "");
    } catch (err) {
      if (err.name !== "AbortError") toast(err.message === "Failed to fetch" ? "网络好像断了，检查一下再试" : err.message);
    } finally {
      el.ai.disabled = false;
      if (state.controller === controller) {
        state.controller = null;
        setBusy(false);
      }
    }
  });

  el.save.addEventListener("click", async () => {
    if (!state.full) return;
    let url = state.resultURL;
    if (state.mode === "local") {
      setBusy(true, "正在生成高清图…");
      el.save.disabled = true;
      try {
        const data = await postForm("/api/retouch", currentFields(state.full, false));
        url = data.image;
      } catch (err) {
        toast(err.message);
        return;
      } finally {
        setBusy(false);
        el.save.disabled = false;
      }
    }
    const blob = await (await fetch(url)).blob();
    if (el.download.href.startsWith("blob:")) URL.revokeObjectURL(el.download.href);
    el.download.href = URL.createObjectURL(blob);
    const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "");
    el.download.download = `seoulglow-${state.style}-${stamp}.jpg`;
    el.saveImg.src = url;
    el.sheet.hidden = false;
    el.closeSheet.focus();
  });

  function closeSheet() { el.sheet.hidden = true; el.save.focus(); }
  el.closeSheet.addEventListener("click", closeSheet);
  el.sheet.addEventListener("click", (e) => { if (e.target === el.sheet) closeSheet(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !el.sheet.hidden) closeSheet(); });

  // ---------------------------------------------------------------- start
  async function init() {
    buildSliders();
    try {
      const resp = await fetch("/api/styles");
      const data = await resp.json();
      state.presets = data.presets;
      state.shades = data.lip_shades;
      state.aiReady = (data.ai_providers || []).length > 0;
      state.maxUploadMb = data.max_upload_mb || 15;
      buildPans();
      buildTints();
      selectStyle(data.default, false);
      el.ai.dataset.locked = String(!state.aiReady);
      el.aiNote.hidden = !state.aiReady;
    } catch (_) {
      toast("连接不上服务器，请确认它已经启动");
    }
  }

  init();
})();
