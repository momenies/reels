/* reels-engine studio — no framework, no build step.
 *
 * Three things drive everything below:
 *   - one `state` object, and `render()` reads it; nothing else touches the DOM
 *   - progress arrives over SSE, with a polling fallback for proxies that
 *     buffer event streams (which many of them do)
 *   - the UI is bilingual, and Arabic flips the whole document to RTL rather
 *     than translating strings into a left-to-right layout
 */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };

  /* ── i18n ──────────────────────────────────────────────────────────── */
  const STRINGS = {
    en: {
      "brand.tag": "long video in, vertical clips out",
      "hero.badge": "engine ready",
      "hero.title": "Turn one long video into <em>clips people finish</em>",
      "hero.sub": "Face-tracked 9:16 reframing, word-level captions that actually shape Arabic, and a model that picks the moments — not the middle.",
      "drop.title": "Drop a video here",
      "drop.sub": "or pick one from your machine — podcasts, interviews, streams, lectures",
      "drop.browse": "Choose a file",
      "drop.types": "MP4 · MOV · MKV · WEBM · AVI · MP3 · WAV",
      "drop.change": "Change",
      "drop.or": "or paste a link",
      "captions.head": "Caption style",
      "settings.head": "Settings",
      "settings.lang": "Spoken language",
      "settings.lang.auto": "Detect automatically",
      "settings.model": "Accuracy",
      "settings.model.default": "Balanced (default)",
      "settings.model.fast": "Fast — small",
      "settings.model.better": "Better — medium",
      "settings.model.best": "Best — large-v3 (GPU)",
      "settings.clips": "How many clips",
      "settings.bar": "Quality bar",
      "settings.bar.hint": "lower it if nothing passes",
      "settings.faces": "Track the speaker",
      "settings.faces.desc": "The crop follows whoever is talking instead of sitting centred",
      "settings.hook": "Burn in a hook title",
      "settings.hook.desc": "Pins the generated headline to the top third of the frame",
      "settings.submit": "Generate clips",
      "progress.head": "Working on it",
      "progress.new": "Start another",
      "clips.head": "Your clips",
      "clips.downloadAll": "Download all",
      "clips.download": "Download",
      "jobs.head": "Recent jobs",
      "jobs.refresh": "Refresh",
      "footer.note": "Source video has to be yours or licensed.",
      "stage.probe": "reading",
      "stage.transcribe": "transcribing",
      "stage.score": "picking moments",
      "stage.render": "rendering",
      "stage.done": "done",
      "status.uploading": "uploading",
      "status.queued": "queued",
      "status.processing": "running",
      "status.done": "done",
      "status.error": "failed",
      "status.canceled": "canceled",
      "empty.clips": "Nothing yet — upload a video to get started.",
      "empty.jobs": "No jobs yet.",
      "toast.uploaded": "Uploaded. The engine has it from here.",
      "toast.ready": "clips are ready",
      "toast.failed": "The job failed",
      "toast.deleted": "Job deleted",
      "toast.nothing": "No moment cleared the quality bar. Try lowering it.",
      "err.pick": "Pick a video or paste a link first.",
      "err.upload": "The upload did not go through",
      "err.network": "Lost contact with the server — retrying",
      "confirm.delete": "Delete this job and its clips?",
      "uploading": "uploading",
    },
    ar: {
      "brand.tag": "فيديو طويل يدخل، مقاطع عمودية تخرج",
      "hero.badge": "المحرك جاهز",
      "hero.title": "حوّل فيديو واحد طويل إلى <em>مقاطع تُشاهد حتى النهاية</em>",
      "hero.sub": "قص عمودي ٩:١٦ يتتبّع الوجه، كابشن يتشكّل بالعربية كما ينبغي، ونموذج يختار اللحظات لا الوسط.",
      "drop.title": "أفلت الفيديو هنا",
      "drop.sub": "أو اختر ملفًا من جهازك — بودكاست، مقابلات، بث، محاضرات",
      "drop.browse": "اختر ملفًا",
      "drop.types": "MP4 · MOV · MKV · WEBM · AVI · MP3 · WAV",
      "drop.change": "تغيير",
      "drop.or": "أو الصق رابطًا",
      "captions.head": "نمط الكابشن",
      "settings.head": "الإعدادات",
      "settings.lang": "لغة الكلام",
      "settings.lang.auto": "اكتشاف تلقائي",
      "settings.model": "الدقة",
      "settings.model.default": "متوازن (افتراضي)",
      "settings.model.fast": "سريع — small",
      "settings.model.better": "أفضل — medium",
      "settings.model.best": "الأعلى — large-v3 (GPU)",
      "settings.clips": "عدد المقاطع",
      "settings.bar": "حدّ الجودة",
      "settings.bar.hint": "أنزله إذا لم يجتز شيء",
      "settings.faces": "تتبّع المتحدث",
      "settings.faces.desc": "يتحرك الإطار خلف من يتكلّم بدل أن يبقى في المنتصف",
      "settings.hook": "اطبع عنوانًا جاذبًا",
      "settings.hook.desc": "يثبّت العنوان المولَّد في الثلث العلوي من الإطار",
      "settings.submit": "أنشئ المقاطع",
      "progress.head": "جارٍ العمل",
      "progress.new": "ابدأ واحدًا آخر",
      "clips.head": "مقاطعك",
      "clips.downloadAll": "تنزيل الكل",
      "clips.download": "تنزيل",
      "jobs.head": "المهام الأخيرة",
      "jobs.refresh": "تحديث",
      "footer.note": "يجب أن يكون الفيديو المصدر ملكك أو مرخّصًا لك.",
      "stage.probe": "قراءة الملف",
      "stage.transcribe": "تفريغ الصوت",
      "stage.score": "اختيار اللحظات",
      "stage.render": "المعالجة",
      "stage.done": "اكتمل",
      "status.uploading": "جارٍ الرفع",
      "status.queued": "في الانتظار",
      "status.processing": "قيد التنفيذ",
      "status.done": "اكتمل",
      "status.error": "فشل",
      "status.canceled": "أُلغي",
      "empty.clips": "لا شيء بعد — ارفع فيديو للبدء.",
      "empty.jobs": "لا توجد مهام بعد.",
      "toast.uploaded": "تم الرفع. المحرك يتولّى الباقي.",
      "toast.ready": "مقاطع جاهزة",
      "toast.failed": "فشلت المهمة",
      "toast.deleted": "حُذفت المهمة",
      "toast.nothing": "لم تجتز أي لحظة حدّ الجودة. جرّب خفضه.",
      "err.pick": "اختر فيديو أو الصق رابطًا أولًا.",
      "err.upload": "لم يكتمل الرفع",
      "err.network": "انقطع الاتصال بالخادم — تجري إعادة المحاولة",
      "confirm.delete": "حذف هذه المهمة ومقاطعها؟",
      "uploading": "جارٍ الرفع",
    },
  };

  const t = (key) => (STRINGS[state.ui] && STRINGS[state.ui][key]) || STRINGS.en[key] || key;

  /* ── state ─────────────────────────────────────────────────────────── */
  const state = {
    ui: localStorage.getItem("reels.ui") || (navigator.language || "en").startsWith("ar") ? "ar" : "en",
    theme: localStorage.getItem("reels.theme") || "dark",
    file: null,
    style: "pop",
    styles: [],
    job: null,
    jobs: [],
    uploading: false,
    urlIngest: false,
    uploadPct: 0,
  };
  // localStorage wins over the browser locale when it holds a real choice.
  const savedUi = localStorage.getItem("reels.ui");
  if (savedUi === "ar" || savedUi === "en") state.ui = savedUi;

  const STAGES = ["probe", "transcribe", "score", "render", "done"];

  /* ── helpers ───────────────────────────────────────────────────────── */
  const fmtBytes = (n) => {
    if (!n) return "—";
    const u = ["B", "KB", "MB", "GB"];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
  };

  const fmtDur = (s) => {
    if (s == null) return "—";
    const m = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    return `${m}:${String(sec).padStart(2, "0")}`;
  };

  const fmtAgo = (ts) => {
    const d = Math.max(0, Date.now() / 1000 - ts);
    if (d < 60) return state.ui === "ar" ? "الآن" : "just now";
    if (d < 3600) return `${Math.floor(d / 60)}m`;
    if (d < 86400) return `${Math.floor(d / 3600)}h`;
    return `${Math.floor(d / 86400)}d`;
  };

  function toast(message, kind = "info") {
    const host = $("#toasts");
    const node = el("div", `toast toast--${kind}`);
    const icons = {
      success: '<path d="M20 6 9 17l-5-5"/>',
      error: '<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/>',
      info: '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/>',
    };
    node.innerHTML =
      `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${icons[kind] || icons.info}</svg>`;
    node.appendChild(el("div", null, message));
    host.appendChild(node);
    setTimeout(() => {
      node.classList.add("is-out");
      setTimeout(() => node.remove(), 260);
    }, 5200);
  }

  async function api(path, options = {}) {
    const res = await fetch(path, options);
    if (!res.ok) {
      let detail = `${res.status}`;
      try {
        const body = await res.json();
        detail = body.detail || detail;
      } catch { /* non-JSON error body */ }
      throw new Error(detail);
    }
    return res.status === 204 ? null : res.json();
  }

  /* ── i18n + theme application ──────────────────────────────────────── */
  function applyLocale() {
    const rtl = state.ui === "ar";
    document.documentElement.lang = state.ui;
    document.documentElement.dir = rtl ? "rtl" : "ltr";
    document.body.dir = rtl ? "rtl" : "ltr";

    document.querySelectorAll("[data-i18n]").forEach((n) => {
      n.textContent = t(n.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-html]").forEach((n) => {
      n.innerHTML = t(n.dataset.i18nHtml);
    });
    document.querySelectorAll("[data-lang-ui]").forEach((b) => {
      b.setAttribute("aria-pressed", String(b.dataset.langUi === state.ui));
    });
    localStorage.setItem("reels.ui", state.ui);
    render();
  }

  function applyTheme() {
    document.documentElement.dataset.theme = state.theme;
    localStorage.setItem("reels.theme", state.theme);
  }

  /* ── rendering ─────────────────────────────────────────────────────── */
  function renderPicked() {
    const has = Boolean(state.file);
    const link = $("#url") ? $("#url").value.trim() : "";
    $("#drop").classList.toggle("has-file", has);
    $("#drop-empty").hidden = has;
    $("#drop-picked").hidden = !has;
    // A link and a file are alternatives; showing both invites sending both.
    if ($("#url-row")) {
      $("#url-row").hidden = has || !state.urlIngest;
      $("#url-field").hidden = has || !state.urlIngest;
    }
    $("#submit").disabled = (!has && !link) || state.uploading;
    if (!has) return;
    $("#picked-name").textContent = state.file.name;
    $("#picked-facts").textContent = fmtBytes(state.file.size);
  }

  function renderStyles() {
    const host = $("#styles");
    host.textContent = "";
    state.styles.forEach((s) => {
      const chip = el("button", "style-chip");
      chip.type = "button";
      chip.setAttribute("aria-pressed", String(s.name === state.style));
      const preview = el("div", "style-chip__preview");
      preview.style.background = s.boxed ? "#000" : "#14131f";
      const sample = el("span", null, s.uppercase ? "WORD" : "Word");
      sample.style.color = s.highlight;
      sample.style.textShadow = "0 0 3px rgba(0,0,0,.9), 0 1px 0 rgba(0,0,0,.9)";
      preview.appendChild(sample);
      chip.appendChild(preview);
      chip.appendChild(el("div", "style-chip__name", s.name));
      chip.title = s.label;
      chip.addEventListener("click", () => {
        state.style = s.name;
        renderStyles();
      });
      host.appendChild(chip);
    });
  }

  function renderProgress() {
    const job = state.job;
    const section = $("#progress-section");
    if (!job && !state.uploading) { section.hidden = true; return; }
    section.hidden = false;

    const uploading = state.uploading;
    const pct = uploading ? state.uploadPct * 0.08 : (job ? job.progress : 0);
    const stage = uploading ? "probe" : (job ? job.stage : "");

    $("#progress-name").textContent = uploading
      ? (state.file ? state.file.name : "")
      : job.filename;
    $("#progress-pct").textContent = `${Math.round(pct * 100)}%`;
    $("#progress-fill").style.width = `${Math.max(2, pct * 100)}%`;
    $("#progress-msg").textContent = uploading
      ? `${t("uploading")} — ${Math.round(state.uploadPct * 100)}%`
      : statusMessage(job);

    const finished = job && (job.status === "done" || job.status === "error");
    $("#progress-bar").classList.toggle("bar--idle", Boolean(finished));

    const host = $("#stages");
    host.textContent = "";
    const reached = STAGES.indexOf(stage);
    STAGES.forEach((name, i) => {
      const node = el("div", "stage");
      if (!finished && i === reached) node.classList.add("is-active");
      else if (finished || (reached > -1 && i < reached)) node.classList.add("is-done");
      node.appendChild(el("span", "stage__dot"));
      node.appendChild(el("span", null, t(`stage.${name}`)));
      host.appendChild(node);
    });

    const alert = $("#progress-alert");
    alert.textContent = "";
    if (job && job.error) {
      const kind = job.status === "error" ? "error" : "warn";
      const box = el("div", `alert alert--${kind}`);
      box.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg>';
      box.appendChild(el("div", null, job.error));
      alert.appendChild(box);
    } else if (job && job.status === "done" && job.clips.length === 0) {
      const box = el("div", "alert alert--warn");
      box.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg>';
      box.appendChild(el("div", null, t("toast.nothing")));
      alert.appendChild(box);
    }
  }

  function statusMessage(job) {
    // The worker writes progress messages in English. For the states the UI
    // can describe itself, say it in the reader's language instead.
    if (job.status === "done") {
      return job.clips.length
        ? `${job.clips.length} ${t("toast.ready")}`
        : t("toast.nothing");
    }
    if (job.status === "error") return t("toast.failed");
    if (job.status === "queued") return t("status.queued");
    if (job.status === "uploading") return t("status.uploading");
    return job.message || "";
  }

  function clipUrl(job, name) { return `/api/jobs/${job.id}/files/${encodeURIComponent(name)}`; }

  function renderClips() {
    const job = state.job;
    const section = $("#clips-section");
    const host = $("#clips-grid");
    if (!job || !job.clips || !job.clips.length) { section.hidden = true; return; }

    section.hidden = false;
    $("#clips-count").textContent = String(job.clips.length);
    host.textContent = "";

    job.clips.forEach((clip, i) => {
      const card = el("article", "clip");
      card.style.animationDelay = `${Math.min(i * 55, 400)}ms`;

      const media = el("div", "clip__media");
      const img = el("img");
      img.loading = "lazy";
      img.alt = clip.title || "";
      img.src = clipUrl(job, clip.thumbnail);
      media.appendChild(img);

      // The clip itself is only fetched on hover — a grid of twelve
      // autoplaying 1080x1920 files is a lot of bandwidth to spend on a
      // preview nobody asked for yet.
      let video = null;
      const startPreview = () => {
        if (!video) {
          video = el("video");
          video.muted = true;
          video.loop = true;
          video.playsInline = true;
          video.preload = "none";
          video.src = clipUrl(job, clip.file);
          media.insertBefore(video, media.firstChild.nextSibling);
        }
        media.classList.add("is-playing");
        video.play().catch(() => media.classList.remove("is-playing"));
      };
      const stopPreview = () => {
        media.classList.remove("is-playing");
        if (video) video.pause();
      };
      media.addEventListener("mouseenter", startPreview);
      media.addEventListener("mouseleave", stopPreview);
      media.addEventListener("click", () => openLightbox(job, clip));

      const score = el("div", "clip__score");
      if (clip.score >= 85) score.classList.add("is-hot");
      else if (clip.score >= 70) score.classList.add("is-good");
      score.textContent = clip.score;
      media.appendChild(score);

      media.appendChild(el("div", "clip__dur", `${Math.round(clip.duration)}s`));

      const play = el("button", "clip__play");
      play.type = "button";
      play.setAttribute("aria-label", clip.title || "play");
      play.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 4l14 8-14 8z"/></svg>';
      media.appendChild(play);

      media.appendChild(el("div", "clip__caption", clip.title || ""));
      card.appendChild(media);

      const body = el("div", "clip__body");
      body.appendChild(el("div", "clip__reason", clip.reason || ""));
      const foot = el("div", "clip__foot");
      const time = el("span", "clip__time", `${fmtDur(clip.start)} → ${fmtDur(clip.end)}`);
      time.dir = "ltr";   // an RTL line would otherwise reorder it to end → start
      foot.appendChild(time);

      const dl = el("a", "btn btn--sm");
      dl.href = clipUrl(job, clip.file);
      dl.download = clip.file;
      dl.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/></svg>';
      foot.appendChild(dl);
      body.appendChild(foot);
      card.appendChild(body);
      host.appendChild(card);
    });
  }

  function renderJobs() {
    const host = $("#jobs");
    const section = $("#jobs-section");
    if (!state.jobs.length) { section.hidden = true; return; }
    section.hidden = false;
    host.textContent = "";

    state.jobs.forEach((job) => {
      const row = el("button", "job");
      row.type = "button";
      if (state.job && job.id === state.job.id) row.classList.add("is-current");
      row.appendChild(el("span", `job__status ${job.status}`));

      const meta = el("div", "job__meta");
      meta.appendChild(el("div", "job__name", job.filename));
      const bits = [t(`status.${job.status}`), fmtAgo(job.created_at)];
      if (job.clips.length) bits.push(`${job.clips.length} ${state.ui === "ar" ? "مقطع" : "clips"}`);
      if (job.source_duration) bits.push(fmtDur(job.source_duration));
      meta.appendChild(el("div", "job__sub", bits.join(" · ")));
      row.appendChild(meta);

      if (job.status === "processing") {
        row.appendChild(el("span", "job__badge", `${Math.round(job.progress * 100)}%`));
      }

      const del = el("span", "btn btn--icon btn--ghost btn--danger");
      del.setAttribute("role", "button");
      del.setAttribute("tabindex", "0");
      del.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>';
      const remove = async (event) => {
        event.stopPropagation();
        if (!confirm(t("confirm.delete"))) return;
        try {
          await api(`/api/jobs/${job.id}`, { method: "DELETE" });
          if (state.job && state.job.id === job.id) state.job = null;
          toast(t("toast.deleted"), "success");
          await loadJobs();
          render();
        } catch (err) {
          toast(err.message, "error");
        }
      };
      del.addEventListener("click", remove);
      row.appendChild(del);

      row.addEventListener("click", () => openJob(job.id));
      host.appendChild(row);
    });
  }

  function render() {
    renderPicked();
    renderStyles();
    renderProgress();
    renderClips();
    renderJobs();
  }

  /* ── lightbox ──────────────────────────────────────────────────────── */
  function openLightbox(job, clip) {
    const box = $("#lightbox");
    const video = $("#lightbox-video");
    video.src = clipUrl(job, clip.file);
    video.poster = clipUrl(job, clip.thumbnail);
    $("#lightbox-title").textContent = clip.title || "";
    $("#lightbox-reason").textContent = clip.reason || "";
    const stats = $("#lightbox-stats");
    stats.textContent = "";
    stats.appendChild(el("span", "pill pill--accent", `${clip.score}/100`));
    stats.appendChild(el("span", "pill", `${Math.round(clip.duration)}s`));
    const range = el("span", "pill", `${fmtDur(clip.start)} → ${fmtDur(clip.end)}`);
    range.dir = "ltr";
    stats.appendChild(range);
    if (clip.bytes) stats.appendChild(el("span", "pill", fmtBytes(clip.bytes)));
    const dl = $("#lightbox-download");
    dl.href = clipUrl(job, clip.file);
    dl.download = clip.file;
    box.hidden = false;
    video.play().catch(() => {});
  }

  function closeLightbox() {
    const video = $("#lightbox-video");
    video.pause();
    video.removeAttribute("src");
    video.load();
    $("#lightbox").hidden = true;
  }

  /* ── job lifecycle ─────────────────────────────────────────────────── */
  let source = null;      // EventSource
  let poller = null;      // setInterval fallback

  function stopWatching() {
    if (source) { source.close(); source = null; }
    if (poller) { clearInterval(poller); poller = null; }
  }

  function onJobUpdate(job, { announce = true } = {}) {
    const wasActive = state.job && (state.job.status === "queued" || state.job.status === "processing");
    state.job = job;
    render();
    if (job.status === "done" && wasActive && announce) {
      if (job.clips.length) toast(`${job.clips.length} ${t("toast.ready")}`, "success");
      else toast(t("toast.nothing"), "info");
      loadJobs().then(render);
    } else if (job.status === "error" && wasActive && announce) {
      toast(`${t("toast.failed")}: ${job.error}`, "error");
      loadJobs().then(render);
    }
  }

  function watch(jobId) {
    stopWatching();

    // SSE is the fast path. Some proxies buffer text/event-stream into
    // uselessness, so a slow poll runs underneath and takes over if the
    // stream never delivers.
    let sawEvent = false;
    try {
      source = new EventSource(`/api/jobs/${jobId}/events`);
      source.onmessage = (e) => {
        sawEvent = true;
        const job = JSON.parse(e.data);
        if (job.error && !job.status) return;
        onJobUpdate(job);
      };
      source.addEventListener("end", () => stopWatching());
      source.onerror = () => { if (source) { source.close(); source = null; } };
    } catch { /* no EventSource: the poller covers it */ }

    poller = setInterval(async () => {
      if (sawEvent && source) return;
      try {
        const job = await api(`/api/jobs/${jobId}`);
        onJobUpdate(job);
        if (job.status !== "queued" && job.status !== "processing") stopWatching();
      } catch (err) {
        // A deleted job is the normal reason this 404s.
        stopWatching();
      }
    }, 2000);
  }

  async function openJob(jobId) {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      state.job = job;
      render();
      $("#progress-section").scrollIntoView({ behavior: "smooth", block: "start" });
      if (job.status === "queued" || job.status === "processing" || job.status === "uploading") watch(job.id);
      else stopWatching();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function loadJobs() {
    try {
      const data = await api("/api/jobs?limit=25");
      state.jobs = data.jobs;
    } catch { /* the history panel is not worth an error toast */ }
  }

  function upload() {
    const link = $("#url") ? $("#url").value.trim() : "";
    if (!state.file && !link) { toast(t("err.pick"), "error"); return; }

    const form = new FormData();
    if (state.file) form.append("file", state.file);
    else form.append("url", link);
    form.append("lang", $("#lang").value);
    form.append("clips", $("#clips").value);
    form.append("min_score", $("#min-score").value);
    form.append("style", state.style);
    form.append("faces", $("#faces").checked ? "true" : "false");
    form.append("hook", $("#hook").checked ? "true" : "false");
    form.append("model", $("#model").value);

    // XHR, not fetch: upload progress on a two-gigabyte file is the
    // difference between "working" and "frozen", and fetch cannot report it.
    const xhr = new XMLHttpRequest();
    state.uploading = true;
    state.uploadPct = 0;
    state.job = null;
    render();

    xhr.upload.addEventListener("progress", (e) => {
      if (!e.lengthComputable) return;
      state.uploadPct = e.loaded / e.total;
      renderProgress();
    });

    xhr.addEventListener("load", async () => {
      state.uploading = false;
      if ($("#url")) $("#url").value = "";
      if (xhr.status === 201) {
        const job = JSON.parse(xhr.responseText);
        state.job = job;
        toast(t("toast.uploaded"), "success");
        await loadJobs();
        render();
        watch(job.id);
      } else {
        let detail = `${xhr.status}`;
        try { detail = JSON.parse(xhr.responseText).detail || detail; } catch {}
        toast(`${t("err.upload")}: ${detail}`, "error");
        render();
      }
    });

    xhr.addEventListener("error", () => {
      state.uploading = false;
      toast(t("err.upload"), "error");
      render();
    });

    xhr.open("POST", "/api/jobs");
    xhr.send(form);
  }

  /* ── wiring ────────────────────────────────────────────────────────── */
  function setFile(file) {
    if (!file) return;
    state.file = file;
    render();
    // A frame from the file itself is a better confirmation than a filename.
    const thumb = $("#picked-thumb");
    if (file.type.startsWith("video/")) {
      const video = el("video");
      video.muted = true;
      video.playsInline = true;
      video.preload = "metadata";
      video.src = URL.createObjectURL(file);
      video.addEventListener("loadedmetadata", () => {
        video.currentTime = Math.min(1, video.duration / 3);
        $("#picked-facts").textContent = `${fmtBytes(file.size)} · ${fmtDur(video.duration)}`;
      }, { once: true });
      thumb.textContent = "";
      thumb.appendChild(video);
    }
  }

  function wire() {
    $("#browse").addEventListener("click", () => $("#file").click());
    $("#file").addEventListener("change", (e) => setFile(e.target.files[0]));
    $("#clear-file").addEventListener("click", () => {
      state.file = null;
      $("#file").value = "";
      $("#picked-thumb").innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="2" y="4" width="20" height="16" rx="2.5"/><path d="m10 9 5 3-5 3V9z" fill="currentColor" stroke="none"/></svg>';
      render();
    });

    const drop = $("#drop");
    ["dragenter", "dragover"].forEach((ev) =>
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
    ["dragleave", "drop"].forEach((ev) =>
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
    drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
    // The whole window accepts a drop, so a file dragged anywhere lands.
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => e.preventDefault());

    $("#submit").addEventListener("click", upload);
    if ($("#url")) $("#url").addEventListener("input", renderPicked);

    $("#clips").addEventListener("input", (e) => { $("#clips-value").textContent = e.target.value; });
    $("#min-score").addEventListener("input", (e) => { $("#min-score-value").textContent = e.target.value; });

    $("#theme-toggle").addEventListener("click", () => {
      state.theme = state.theme === "dark" ? "light" : "dark";
      applyTheme();
    });

    document.querySelectorAll("[data-lang-ui]").forEach((b) =>
      b.addEventListener("click", () => { state.ui = b.dataset.langUi; applyLocale(); }));

    $("#progress-hide").addEventListener("click", () => {
      stopWatching();
      state.job = null;
      state.file = null;
      $("#file").value = "";
      render();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });

    $("#jobs-refresh").addEventListener("click", async () => { await loadJobs(); render(); });

    $("#download-all").addEventListener("click", () => {
      if (state.job) window.location.href = `/api/jobs/${state.job.id}/download`;
    });

    $("#lightbox-close").addEventListener("click", closeLightbox);
    $("#lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox") closeLightbox(); });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !$("#lightbox").hidden) closeLightbox();
    });
  }

  async function boot() {
    applyTheme();
    wire();
    applyLocale();

    try {
      const health = await api("/api/health");
      state.urlIngest = Boolean(health.url_ingest);
      if (!health.ffmpeg) {
        $("#health-dot").style.background = "var(--danger)";
        $("#health-text").textContent = "ffmpeg missing";
      }
    } catch {
      $("#health-dot").style.background = "var(--danger)";
      $("#health-text").textContent = "offline";
    }

    try {
      const data = await api("/api/styles");
      state.styles = data.styles;
    } catch { /* the picker just stays empty */ }

    await loadJobs();
    render();

    // Reattach to whatever is still running after a refresh.
    const live = state.jobs.find((j) => j.status === "queued" || j.status === "processing" || j.status === "uploading");
    if (live) openJob(live.id);
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
