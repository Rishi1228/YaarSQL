// HMM POS Tagger page.
const tagClass = (t) => `pos-${t === "." ? "PUNCT" : t}`;
let modelData, evalData, f1Chart, mixChart, evalLang = "en";

function currentModelArgs(lang) {
  const l = lang || $("pos-lang").value || "hg";
  return { lang: l, mix: l === "hg" ? $("pos-mix").value : "" };
}

async function runTagger() {
  $("pos-error").hidden = true;
  const text = $("pos-text").value.trim();
  if (!text) return;
  let d;
  try {
    d = await api("/api/pos/tag", { text, lang: $("pos-lang").value, mix: $("pos-mix").value });
  } catch (e) {
    $("pos-error").textContent = e.message;
    $("pos-error").hidden = false;
    return;
  }
  $("tagged").innerHTML = d.columns
    .map(
      (c) => `<span class="pos-tok ${tagClass(c.tag)}${c.known ? "" : " unknown"}" title="${escapeHtml(d.tag_info[c.tag] || c.tag)}${c.known ? "" : " · unseen word"}">
        <span class="w">${escapeHtml(c.word)}</span><span class="t">${c.tag}</span></span>`
    )
    .join("");
  $("pos-stats").innerHTML = `
    <span>Detected language: <b>${LANG_NAMES[d.lang === "hi-iiit" ? "hi" : d.lang]}</b></span>
    <span>log P(best tag path, words) = <b>${fmt(d.viterbi_logprob, 2)}</b> (Viterbi)</span>
    <span>log P(words) = <b>${fmt(d.forward_logprob, 2)}</b> (forward algorithm)</span>
    <span class="muted">dashed border = word not seen in training</span>`;
  drawTrellis(d);
  const { lang, mix } = { lang: d.lang, mix: $("pos-mix").value };
  loadModel(lang, mix);
}

function drawTrellis(d) {
  const head = `<tr><th></th>${d.columns.map((c) => `<th class="${c.known ? "" : "unk-col"}">${escapeHtml(c.word)}</th>`).join("")}</tr>`;
  const rows = d.tags
    .map((t) => {
      const cells = d.columns
        .map((c, i) => {
          const v = c.scores[t];
          const onPath = c.tag === t;
          const a = Math.max(0, 1 + v / 25); // fade with distance from the best score
          const back = c.back[t] ? `<small>${c.back[t] === "." ? "·" : c.back[t].slice(0, 2)}</small>` : "";
          return `<td class="${onPath ? "on-path" : ""}" style="${onPath ? "" : `background: rgba(61,90,254,${(a * 0.22).toFixed(3)})`}">${v === 0 ? "0" : v.toFixed(1)}${i > 0 ? back : ""}</td>`;
        })
        .join("");
      return `<tr><th><span class="tag ${tagClass(t)}">${t}</span></th>${cells}</tr>`;
    })
    .join("");
  $("trellis").innerHTML = head + rows;
}

async function loadModel(lang, mix) {
  modelData = await api(`/api/pos/model?${new URLSearchParams({ lang, mix: mix || "" })}`);
  const tm = modelData.transitions;
  const head = `<tr><th>prev \\ next</th>${tm.cols.map((c) => `<th>${c === "</S>" ? "END" : c}</th>`).join("")}</tr>`;
  const body = tm.rows
    .map(
      (r, i) =>
        `<tr><th>${r === "<S>" ? "START" : r}</th>${tm.p[i]
          .map((p) => {
            const a = Math.sqrt(p);
            return `<td style="background: rgba(61,90,254,${(a * 0.85).toFixed(3)}); color:${a > 0.55 ? "#fff" : "inherit"}">${p < 0.005 ? "" : p.toFixed(2).replace(/^0/, "")}</td>`;
          })
          .join("")}</tr>`
    )
    .join("");
  $("trans").innerHTML = head + body;

  const sel = $("emit-tag");
  const prev = sel.value || "NOUN";
  const tags = Object.keys(modelData.emissions);
  sel.innerHTML = tags.map((t) => `<option value="${t}"${t === prev ? " selected" : ""}>${t} — ${modelData.tag_info[t] || t}</option>`).join("");
  $("model-meta").textContent = `Trained on ${modelData.train_tokens.toLocaleString()} tagged tokens, vocabulary ${modelData.vocab.toLocaleString()} words.`;
  drawEmissions();
}

function drawEmissions() {
  const t = $("emit-tag").value;
  const rows = modelData.emissions[t] || [];
  const max = Math.max(...rows.map((r) => r.p), 1e-9);
  $("emissions").innerHTML = rows
    .map((r) => `<div class="bar-row"><span>${escapeHtml(r.word)}</span><div class="bar"><div style="width:${((100 * r.p) / max).toFixed(1)}%"></div></div><span class="num small">${r.p.toFixed(4)}</span></div>`)
    .join("");
}

// ---------------- evaluation ----------------
async function loadEval() {
  evalData = await api("/api/pos/eval");
  const L = evalData.languages;
  $("eval-cards").innerHTML = Object.entries(L)
    .map(
      ([k, r]) => `<div class="stat"><span class="stat-label">${r.label}</span><span class="stat-value">${pct(r.hmm.accuracy)}</span>
        <span class="muted small">known ${pct(r.hmm.known_accuracy)} · unseen ${r.hmm.unknown_accuracy == null ? "–" : pct(r.hmm.unknown_accuracy)}</span></div>`
    )
    .join("");

  const baselineNames = [...new Set(Object.values(L).flatMap((r) => Object.keys(r.baselines)))];
  $("baseline-table").innerHTML =
    `<thead><tr><th>Language</th><th class="num">Train / test sents</th>${baselineNames.map((b) => `<th class="num">${b}</th>`).join("")}<th class="num">Our HMM</th></tr></thead><tbody>` +
    Object.values(L)
      .map((r) => {
        const vals = [...baselineNames.map((b) => r.baselines[b]), r.hmm.accuracy];
        const best = Math.max(...vals);
        return `<tr><td>${r.label}</td><td class="num">${r.train_sentences} / ${r.test_sentences}</td>${vals
          .map((v) => `<td class="num${v === best ? " best" : ""}">${pct(v)}</td>`)
          .join("")}</tr>`;
      })
      .join("") +
    "</tbody>";
  const cv = L.hg && L.hg.cv;
  if (cv) {
    $("cv-note").innerHTML = `Hinglish test split is small (${L.hg.hmm.tokens} words), so we also report 5-fold cross-validation: ` +
      Object.entries(cv).map(([k, v]) => `${k} <b>${pct(v)}</b>`).join(" vs ") + ".";
  }

  const mixes = Object.entries(evalData.hinglish_mixes);
  if (mixChart) mixChart.destroy();
  mixChart = new Chart($("mix-chart"), {
    type: "bar",
    data: {
      labels: mixes.map(([, r]) => r.label),
      datasets: [
        { label: "Test split", data: mixes.map(([, r]) => 100 * r.accuracy), backgroundColor: "#b9c4ff", borderRadius: 3 },
        { label: "5-fold CV", data: mixes.map(([, r]) => 100 * r.cv_accuracy), backgroundColor: cssVar("--accent"), borderRadius: 3 },
      ],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } },
      scales: { x: { min: 40, max: 100, title: { display: true, text: "accuracy %" }, grid: { color: "#eef0f3" } }, y: { grid: { display: false }, ticks: { font: { size: 11 } } } },
    },
  });

  $("eval-tabs").innerHTML = Object.entries(L)
    .map(([k, r]) => `<button class="tab${k === evalLang ? " active" : ""}" data-lang="${k}">${r.label.replace(/ \(.*\)/, k === "hi-iiit" ? " IIIT" : "")}</button>`)
    .join("");
  $("eval-tabs").querySelectorAll("[data-lang]").forEach((b) =>
    b.addEventListener("click", () => {
      evalLang = b.dataset.lang;
      $("eval-tabs").querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
      drawPerTag();
    })
  );
  drawPerTag();
}

function drawPerTag() {
  const h = evalData.languages[evalLang].hmm;
  const tags = h.tags.filter((t) => h.per_tag[t].support > 0);
  if (f1Chart) f1Chart.destroy();
  f1Chart = new Chart($("f1-chart"), {
    type: "bar",
    data: {
      labels: tags.map((t) => `${t} (${h.per_tag[t].support})`),
      datasets: [
        { label: "Precision", data: tags.map((t) => h.per_tag[t].precision), backgroundColor: "#b9c4ff", borderRadius: 3 },
        { label: "Recall", data: tags.map((t) => h.per_tag[t].recall), backgroundColor: "#7c8cff", borderRadius: 3 },
        { label: "F1", data: tags.map((t) => h.per_tag[t].f1), backgroundColor: cssVar("--accent"), borderRadius: 3 },
      ],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } },
      scales: { x: { min: 0, max: 1, grid: { color: "#eef0f3" } }, y: { grid: { display: false } } },
    },
  });
  const max = Math.max(...h.confusion.flat());
  $("confusion").innerHTML =
    `<tr><th></th>${h.tags.map((t) => `<th>${t}</th>`).join("")}</tr>` +
    h.tags
      .map(
        (g, i) =>
          `<tr><th>${g}</th>${h.confusion[i]
            .map((v, j) => {
              const a = v ? Math.sqrt(v / max) : 0;
              const col = i === j ? "61,90,254" : "192,54,44";
              return `<td style="background: rgba(${col},${(a * 0.8).toFixed(3)}); color:${a > 0.55 ? "#fff" : "inherit"}">${v || ""}</td>`;
            })
            .join("")}</tr>`
      )
      .join("");
}

$("pos-run").addEventListener("click", runTagger);
$("pos-text").addEventListener("keydown", (e) => e.key === "Enter" && runTagger());
$("pos-lang").addEventListener("change", () => {
  $("mix-field").style.display = ["", "hg"].includes($("pos-lang").value) ? "" : "none";
  runTagger();
});
$("pos-mix").addEventListener("change", runTagger);
$("emit-tag").addEventListener("change", drawEmissions);
document.querySelectorAll("[data-text]").forEach((b) =>
  b.addEventListener("click", () => {
    $("pos-text").value = b.dataset.text;
    $("pos-lang").value = "";
    $("mix-field").style.display = "";
    runTagger();
  })
);
runTagger();
loadEval();
