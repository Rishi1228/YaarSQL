// Evaluation page.
let R, langChart, ablChart;
const LANGS = ["en", "hi", "hg"];
const LCOL = { en: "#2563eb", hi: "#c2410c", hg: "#7c3aed" };

function render() {
  const T = R.splits.test.summary;
  const W = R.splits.wild ? R.splits.wild.summary : null;
  $("gen").textContent = `Results generated ${R.generated}. Re-running takes about a minute.`;

  $("cards").innerHTML =
    `<div class="stat accent"><span class="stat-label">Test · all languages</span><span class="stat-value">${pct(T.all.execution_accuracy)}</span><span class="muted small">${T.all.n} questions · exact match ${pct(T.all.exact_match)}</span></div>` +
    LANGS.map((l) => `<div class="stat"><span class="stat-label">Test · ${LANG_NAMES[l]}</span><span class="stat-value">${pct(T[l].execution_accuracy)}</span><span class="muted small">${T[l].n} questions</span></div>`).join("") +
    (W ? `<div class="stat"><span class="stat-label">Wild · hand-written</span><span class="stat-value">${pct(W.all.execution_accuracy)}</span><span class="muted small">${LANGS.map((l) => `${l} ${pct(W[l].execution_accuracy)}`).join(" · ")}</span></div>` : "");

  if (langChart) langChart.destroy();
  langChart = new Chart($("lang-chart"), {
    type: "bar",
    data: {
      labels: LANGS.map((l) => LANG_NAMES[l]),
      datasets: [
        { label: "Test · execution", data: LANGS.map((l) => 100 * T[l].execution_accuracy), backgroundColor: cssVar("--accent"), borderRadius: 4 },
        { label: "Test · exact match", data: LANGS.map((l) => 100 * T[l].exact_match), backgroundColor: "#b9c4ff", borderRadius: 4 },
        ...(W ? [{ label: "Wild · execution", data: LANGS.map((l) => 100 * W[l].execution_accuracy), backgroundColor: "#12805c", borderRadius: 4 }] : []),
      ],
    },
    options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } },
      scales: { y: { min: 0, max: 100, title: { display: true, text: "%" }, grid: { color: "#eef0f3" } }, x: { grid: { display: false } } } },
  });

  const A = Object.entries(R.ablations);
  if (ablChart) ablChart.destroy();
  ablChart = new Chart($("abl-chart"), {
    type: "bar",
    data: {
      labels: A.map(([, a]) => a.label),
      datasets: LANGS.map((l) => ({ label: LANG_NAMES[l], data: A.map(([, a]) => 100 * a[l].execution_accuracy), backgroundColor: LCOL[l], borderRadius: 3 })),
    },
    options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } },
      scales: { x: { min: 0, max: 100, grid: { color: "#eef0f3" } }, y: { grid: { display: false }, ticks: { font: { size: 11 } } } } },
  });

  const full = R.ablations.full.all.execution_accuracy;
  $("abl-table").innerHTML =
    `<thead><tr><th>Configuration</th>${LANGS.map((l) => `<th class="num">${l}</th>`).join("")}<th class="num">All</th><th class="num">Δ</th><th class="num">Wild</th></tr></thead><tbody>` +
    A.map(([k, a]) => {
      const d = a.all.execution_accuracy - full;
      return `<tr${k === "full" ? ' class="strong"' : ""}><td>${a.label}</td>${LANGS.map((l) => `<td class="num">${pct(a[l].execution_accuracy)}</td>`).join("")}
        <td class="num"><b>${pct(a.all.execution_accuracy)}</b></td><td class="num ${d < -0.0005 ? "neg" : d > 0.0005 ? "pos" : ""}">${k === "full" ? "" : (d >= 0 ? "+" : "") + (100 * d).toFixed(1)}</td>
        <td class="num">${a.wild != null ? pct(a.wild) : "–"}</td></tr>`;
    }).join("") + "</tbody>";
  if (R.validation) {
    const v = R.validation;
    $("val-note").innerHTML = `Settings were chosen on a <b>validation</b> phrasing held out from the training data, never on the test set.
      Validation accuracy: full ${pct(v.full)}, without reasoning ${pct(v.no_reasoning)}, unigram intents ${pct(v.unigram_intent)}, trigram intents ${pct(v.trigram_intent)}.
      Unigram intents score higher on test but lower on validation, so the bigram default was kept.`;
  }

  const comps = ["table", "aggregation", "conditions", "group_by", "order", "limit"];
  $("comp-table").innerHTML =
    `<thead><tr><th>Clause</th>${LANGS.map((l) => `<th class="num">${LANG_NAMES[l]}</th>`).join("")}</tr></thead><tbody>` +
    comps.map((c) => `<tr><td>${c.replace("_", " ")}</td>${LANGS.map((l) => `<td class="num">${pct(T[l].components[c])}</td>`).join("")}</tr>`).join("") + "</tbody>";

  const E = R.splits.test.errors;
  const cats = [...new Set(LANGS.flatMap((l) => Object.keys(E[l] || {})))];
  $("err-table").innerHTML = cats.length
    ? `<thead><tr><th>Error type</th>${LANGS.map((l) => `<th class="num">${l}</th>`).join("")}</tr></thead><tbody>` +
      cats.map((c) => `<tr><td>${c}</td>${LANGS.map((l) => `<td class="num">${(E[l] || {})[c] || 0}</td>`).join("")}</tr>`).join("") + "</tbody>"
    : "<tr><td>No errors.</td></tr>";

  $("lid-table").innerHTML = `<thead><tr><th>Set</th>${LANGS.map((l) => `<th class="num">${l}</th>`).join("")}</tr></thead><tbody>` +
    Object.entries(R.splits).map(([s, r]) => `<tr><td>${s}</td>${LANGS.map((l) => `<td class="num">${pct(r.summary[l].language_id)}</td>`).join("")}</tr>`).join("") + "</tbody>";
  $("intent-table").innerHTML = `<thead><tr><th>Model</th><th class="num">Top-1</th><th class="num">Top-3</th></tr></thead><tbody>` +
    Object.entries(R.intent).map(([n, r]) => `<tr><td>${{ 1: "Unigram (Naive Bayes)", 2: "Bigram", 3: "Trigram" }[n]}</td><td class="num">${pct(r.all.top1)}</td><td class="num">${pct(r.all.top3)}</td></tr>`).join("") +
    `</tbody><caption class="muted small">On its own, before candidate selection fixes most mistakes.</caption>`;
  $("pos-table").innerHTML = `<thead><tr><th>Corpus</th><th class="num">Accuracy</th></tr></thead><tbody>` +
    Object.values(R.pos).map((p) => `<tr><td>${p.label}</td><td class="num">${pct(p.accuracy)}</td></tr>`).join("") + "</tbody>";

  renderExamples();
}

function renderExamples() {
  const split = R.splits[$("ex-split").value];
  if (!split) return;
  const lang = $("ex-lang").value;
  const rows = split.examples.filter((e) => (!lang || e.lang === lang) && (!$("ex-wrong").checked || !e.exec));
  $("ex-table").innerHTML =
    `<thead><tr><th></th><th>Lang</th><th>Question</th><th>Gold SQL</th><th>Predicted SQL</th></tr></thead><tbody>` +
    rows.slice(0, 500).map((e) => `<tr class="${e.exec ? "" : "wrong-row"}"><td>${e.exec ? "✓" : "✗"}</td><td>${e.lang}</td><td>${escapeHtml(e.question)}</td>
      <td class="mono small">${escapeHtml(e.gold)}</td><td class="mono small">${escapeHtml(e.pred || "—")}</td></tr>`).join("") +
    `</tbody>`;
}

async function load() {
  try {
    R = await api("/api/eval");
    render();
  } catch (e) {
    $("err").textContent = e.message;
    $("err").hidden = false;
  }
}

$("rerun").addEventListener("click", async () => {
  $("rerun").disabled = true;
  $("rerun").textContent = "Running… (about a minute)";
  try {
    R = await api("/api/eval/run", {});
    render();
  } catch (e) {
    $("err").textContent = e.message;
    $("err").hidden = false;
  } finally {
    $("rerun").disabled = false;
    $("rerun").textContent = "Re-run evaluation";
  }
});
["ex-split", "ex-lang", "ex-wrong"].forEach((id) => $(id).addEventListener("change", renderExamples));
load();
