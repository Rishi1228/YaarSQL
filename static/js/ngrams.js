// N-gram Lab page.
const SERIES_COLORS = ["#9aa3b2", "#e8590c", "#7c3aed", "#0f766e", "#c2255c", "#3d5afe"];
let topChart, pplChart, overviewData, matrixMode = "probs", currentOrder = 1;

function params() {
  return { corpus: $("corpus").value, n: $("n").value, smoothing: $("smoothing").value };
}

async function loadOverview() {
  $("status").textContent = "Training…";
  const p = params();
  try {
    overviewData = await api(`/api/ngram/overview?${new URLSearchParams(p)}`);
  } catch (e) {
    $("status").textContent = e.message;
    return;
  }
  $("status").textContent = "";
  const s = overviewData.stats;
  $("st-sent").textContent = s.sentences.toLocaleString();
  $("st-tok").textContent = s.tokens.toLocaleString();
  $("st-vocab").textContent = s.vocab.toLocaleString();
  $("st-ppl").textContent = fmt(overviewData.test_perplexity, 1);

  const orders = Object.keys(overviewData.top).map(Number);
  if (!orders.includes(currentOrder)) currentOrder = orders[orders.length - 1];
  const names = { 1: "Unigrams", 2: "Bigrams", 3: "Trigrams", 4: "4-grams" };
  $("order-tabs").innerHTML = orders
    .map((o) => `<button class="tab${o === currentOrder ? " active" : ""}" data-order="${o}">${names[o]}</button>`)
    .join("");
  $("order-tabs").querySelectorAll("[data-order]").forEach((b) =>
    b.addEventListener("click", () => {
      currentOrder = Number(b.dataset.order);
      $("order-tabs").querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === b));
      drawTop();
    })
  );
  drawTop();
  drawMatrix();
  $("generated").innerHTML = overviewData.generated
    .map((g) => `<li>${g ? escapeHtml(g) : '<span class="muted">(empty sentence)</span>'}</li>`)
    .join("");
  scoreSentence();
}

function drawTop() {
  const rows = overviewData.top[currentOrder];
  const data = {
    labels: rows.map((r) => r.gram),
    datasets: [{ data: rows.map((r) => r.count), backgroundColor: cssVar("--accent"), borderRadius: 4, maxBarThickness: 18 }],
  };
  if (topChart) topChart.destroy();
  topChart = new Chart($("top-chart"), {
    type: "bar",
    data,
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` count: ${c.raw}` } } },
      scales: { x: { grid: { color: "#eef0f3" }, title: { display: true, text: "count" } }, y: { grid: { display: false }, ticks: { font: { size: 12 } } } },
    },
  });
}

function drawMatrix() {
  const m = overviewData.matrix;
  const vals = m[matrixMode];
  const max = Math.max(...vals.flat(), 1e-9);
  const head = `<tr><th>w₁ \\ w₂</th>${m.words.map((w) => `<th>${escapeHtml(w)}</th>`).join("")}</tr>`;
  const body = m.words
    .map((w, i) => {
      const cells = vals[i]
        .map((v) => {
          const a = Math.sqrt(v / max); // sqrt so small values still show
          const txt = matrixMode === "probs" ? (v < 0.001 ? "<.001" : v.toFixed(3).replace(/^0/, "")) : v;
          return `<td style="background: rgba(61,90,254,${(a * 0.75).toFixed(3)}); color: ${a > 0.6 ? "#fff" : "inherit"}">${txt}</td>`;
        })
        .join("");
      return `<tr><th>${escapeHtml(w)}</th>${cells}</tr>`;
    })
    .join("");
  $("matrix").innerHTML = head + body;
}

document.querySelectorAll("[data-matrix]").forEach((b) =>
  b.addEventListener("click", () => {
    matrixMode = b.dataset.matrix;
    document.querySelectorAll("[data-matrix]").forEach((t) => t.classList.toggle("active", t === b));
    drawMatrix();
  })
);

async function loadPerplexity() {
  $("ppl-note").textContent = "Training 24 models…";
  const d = await api(`/api/ngram/perplexity?corpus=${$("corpus").value}`);
  $("ppl-note").textContent = `${d.test_sentences} held-out sentences`;
  const keys = Object.keys(d.table);
  if (pplChart) pplChart.destroy();
  pplChart = new Chart($("ppl-chart"), {
    type: "line",
    data: {
      labels: d.orders.map((o) => `N=${o}`),
      datasets: keys
        .filter((k) => k !== "mle")
        .map((k, i) => ({
          label: d.labels[k],
          data: d.table[k],
          borderColor: SERIES_COLORS[i + 1],
          backgroundColor: SERIES_COLORS[i + 1],
          borderDash: k === "backoff" ? [5, 4] : [],
          tension: 0.25,
        })),
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } },
      scales: { y: { title: { display: true, text: "perplexity" }, grid: { color: "#eef0f3" } }, x: { grid: { display: false } } },
    },
  });
  const best = Math.min(...keys.filter((k) => k !== "backoff").flatMap((k) => d.table[k].filter((v) => v !== null)));
  $("ppl-table").innerHTML =
    `<thead><tr><th>Smoothing</th>${d.orders.map((o) => `<th class="num">N=${o}</th>`).join("")}</tr></thead><tbody>` +
    keys
      .map((k) => `<tr><td>${d.labels[k]}</td>${d.table[k].map((v) => `<td class="num${v === best ? " best" : ""}">${fmt(v, 1)}</td>`).join("")}</tr>`)
      .join("") +
    "</tbody>";
}

async function scoreSentence() {
  const text = $("sent-text").value.trim();
  if (!text) return;
  try {
    const d = await api("/api/ngram/sentence", { ...params(), text });
    const rows = d.rows
      .map(
        (r) => `<tr class="${r.known ? "" : "unk-row"}"><td class="muted">${escapeHtml(r.context.join(" ") || "∅")}</td>
          <td><b>${escapeHtml(r.word)}</b>${r.known ? "" : ' <span class="tag lang-UNK">unseen</span>'}</td>
          <td class="num">${fmt(r.p, 4)}</td><td class="num">${fmt(r.logp, 2)}</td></tr>`
      )
      .join("");
    $("sent-result").innerHTML = `
      <table class="data-table compact"><thead><tr><th>Context</th><th>Word</th><th class="num">P(word | context)</th><th class="num">log P</th></tr></thead>
      <tbody>${rows}</tbody></table>
      <div class="kv"><span>log P(sentence) = <b>${fmt(d.logprob, 2)}</b></span><span>P = <b>${fmt(d.prob, 4)}</b></span><span>perplexity = <b>${fmt(d.perplexity, 1)}</b></span></div>`;
  } catch (e) {
    $("sent-result").innerHTML = `<p class="error">${escapeHtml(e.message)}</p>`;
  }
}

// ---------------- project uses ----------------
const autocomplete = debounce(async () => {
  const d = await api(`/api/autocomplete?text=${encodeURIComponent($("ac-text").value)}`);
  $("ac-out").innerHTML =
    `<p class="muted small">Language: ${LANG_NAMES[d.lang]} · context: <b>${escapeHtml(d.context.join(" ") || "∅")}</b>${d.prefix ? ` · completing “${escapeHtml(d.prefix)}”` : ""}</p>` +
    (d.suggestions.length
      ? d.suggestions.map((s) => `<button class="suggestion" data-w="${escapeHtml(s.word)}">${escapeHtml(s.word === "</s>" ? "⏎ end" : s.word)} <small>${fmt(s.p, 3)}</small></button>`).join("")
      : '<span class="muted">No suggestions.</span>');
  $("ac-out").querySelectorAll("[data-w]").forEach((b) =>
    b.addEventListener("click", () => {
      if (b.dataset.w === "</s>") return;
      const t = $("ac-text").value;
      const base = t.endsWith(" ") || !t ? t : t.slice(0, t.lastIndexOf(" ") + 1);
      $("ac-text").value = base + b.dataset.w + " ";
      $("ac-text").focus();
      autocomplete();
    })
  );
}, 150);

async function detectLang() {
  try {
    const d = await api("/api/langid", { text: $("lid-text").value });
    $("lid-out").innerHTML =
      `<p class="result-line">Detected: <b>${LANG_NAMES[d.prediction]}</b></p>` +
      Object.keys(d.posterior)
        .map(
          (l) => `<div class="bar-row"><span>${LANG_NAMES[l]}</span><div class="bar"><div style="width:${(100 * d.posterior[l]).toFixed(1)}%"></div></div>
            <span class="num small">log P ${fmt(d.logp[l], 1)}</span></div>`
        )
        .join("");
  } catch (e) {
    $("lid-out").innerHTML = `<p class="error">${escapeHtml(e.message)}</p>`;
  }
}

async function classify() {
  try {
    const d = await api("/api/intent", { text: $("int-text").value });
    $("int-out").innerHTML =
      `<p class="muted small">Model input: ${d.features.map((w) => `<span class="tok mini">${escapeHtml(w)}</span>`).join(" ")}</p>` +
      d.ranking
        .map(
          (r, i) => `<div class="bar-row"><span class="${i === 0 ? "strong" : ""}">${r.intent}</span><div class="bar"><div style="width:${(100 * r.prob).toFixed(1)}%"></div></div>
            <span class="num small">${pct(r.prob)}</span></div>`
        )
        .join("");
  } catch (e) {
    $("int-out").innerHTML = `<p class="error">${escapeHtml(e.message)}</p>`;
  }
}

["corpus", "n", "smoothing"].forEach((id) => $(id).addEventListener("change", loadOverview));
$("corpus").addEventListener("change", loadPerplexity);
$("sent-run").addEventListener("click", scoreSentence);
$("sent-text").addEventListener("keydown", (e) => e.key === "Enter" && scoreSentence());
$("ac-text").addEventListener("input", autocomplete);
$("lid-run").addEventListener("click", detectLang);
$("lid-text").addEventListener("keydown", (e) => e.key === "Enter" && detectLang());
$("int-run").addEventListener("click", classify);
$("int-text").addEventListener("keydown", (e) => e.key === "Enter" && classify());

loadOverview();
loadPerplexity();
autocomplete();
detectLang();
classify();
