// Ask the Database page.
const tagClass = (t) => `pos-${t === "." ? "PUNCT" : t}`;

function renderResult(res) {
  if (!res) return "";
  if (res.error) return `<p class="error">${escapeHtml(res.error)}</p>`;
  const { columns, rows } = res;
  if (rows.length === 1 && columns.length === 1) {
    const v = rows[0][0];
    const shown = typeof v === "number" && !Number.isInteger(v) ? v.toFixed(2) : v;
    return `<div class="big-answer">${escapeHtml(shown ?? "—")}</div><p class="muted small">${escapeHtml(columns[0])}</p>`;
  }
  if (!rows.length) return '<p class="muted">No rows match.</p>';
  const head = `<tr><th>#</th>${columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr>`;
  const body = rows
    .map((r, i) => `<tr><td class="muted">${i + 1}</td>${r.map((v) => `<td>${escapeHtml(typeof v === "number" && !Number.isInteger(v) ? v.toFixed(2) : v)}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="result-scroll"><table class="data-table compact">${head}${body}</table></div>${res.truncated ? '<p class="muted small">Showing the first 100 rows.</p>' : ""}`;
}

function highlightSql(sql) {
  return escapeHtml(sql)
    .replace(/\b(SELECT|FROM|JOIN|ON|WHERE|AND|GROUP BY|ORDER BY|LIMIT|DESC|ASC)\b/g, '<span class="kw">$1</span>')
    .replace(/\b(COUNT|MAX|MIN|AVG|SUM)\b/g, '<span class="fn">$1</span>')
    .replace(/(&#39;[^&]*&#39;)/g, '<span class="str">$1</span>')
    .replace(/ (FROM|JOIN|WHERE|GROUP BY|ORDER BY|LIMIT)/g, "\n$1");
}

function stage(n, title, desc, body) {
  return `<div class="stage"><div class="stage-num">${n}</div><div class="stage-body"><h3>${title}</h3><p class="muted small">${desc}</p>${body}</div></div>`;
}

function renderPipeline(t) {
  const langBars = Object.entries(t.lang_scores)
    .map(([l, p]) => `<div class="bar-row"><span>${LANG_NAMES[l]}</span><div class="bar"><div style="width:${(100 * p).toFixed(1)}%"></div></div><span class="num small">${pct(p)}</span></div>`)
    .join("");
  const fixes = t.spelling_changes.length
    ? `<p class="small">Spelling fixed: ${t.spelling_changes.map((c) => `<span class="fix"><s>${escapeHtml(c.from)}</s> → <b>${escapeHtml(c.to)}</b></span>`).join(" ")}</p>`
    : "";
  const pos = t.pos.map((p) => `<span class="pos-tok ${tagClass(p.tag)}"><span class="w">${escapeHtml(p.word)}</span><span class="t">${p.tag}</span></span>`).join("");
  const links = t.mentions.length
    ? t.mentions
        .map((m) => {
          const target = m.type === "number" ? m.value : m.type === "value" ? `${m.key} = '${m.value}'` : m.key;
          return `<span class="link-chip type-${m.type}"><b>${escapeHtml(m.text)}</b> → ${escapeHtml(target)}<small>${m.type}${m.fuzzy ? " · fuzzy" : ""}</small></span>`;
        })
        .join("")
    : '<span class="muted">Nothing linked.</span>';
  const nums = t.numbers.length ? `<p class="small">Numbers: ${t.numbers.map((n) => `<b>${n.value}</b> = ${n.role}`).join(", ")}</p>` : "";
  const intents = t.intents
    .slice(0, 5)
    .map((r) => `<div class="bar-row"><span class="small">${r.intent}</span><div class="bar"><div style="width:${(100 * r.prob).toFixed(1)}%"></div></div><span class="num small">${pct(r.prob)}</span></div>`)
    .join("");
  const valid = t.candidates.filter((c) => c.sql);
  const invalid = t.candidates.filter((c) => !c.sql);
  const cands = valid
    .slice(0, 6)
    .map(
      (c) => `<tr class="${c.chosen ? "chosen" : ""}">
        <td>${c.chosen ? "✓ " : ""}${c.intent}</td>
        <td class="mono small">${escapeHtml(c.sql)}</td>
        <td class="num">${pct(c.intent_prob)}</td>
        <td class="num">${c.similarity.toFixed(2)}</td>
        <td class="small">${c.penalties.map((p) => `<div>− ${escapeHtml(p)}</div>`).join("") || '<span class="muted">none</span>'}</td>
        <td class="num"><b>${fmt(c.score, 2)}</b></td></tr>`
    )
    .join("");
  const chosen = valid.find((c) => c.chosen);
  const frags = chosen ? chosen.fragments.map((f) => `<span class="frag mono">${escapeHtml(f)}</span>`).join("") : "";

  $("pipeline").innerHTML = [
    stage(1, "Language identification", "Character-trigram language models (Phase 2).", langBars),
    stage(2, "Preprocessing", "Normalization, tokenization, Hinglish spelling repair (Phase 1).", fixes + `<div class="tokens-line">${t.tokens.map((w) => `<span class="tok">${escapeHtml(w)}</span>`).join("")}</div>`),
    stage(3, "POS tagging", "HMM + Viterbi for the detected language (Phase 3). Nouns guide schema linking.", `<div class="tagged">${pos}</div>`),
    stage(4, "Schema linking", "Words matched to tables, columns and values in English, Hindi and Hinglish.", links + nums),
    stage(5, "Question type", "One bigram model per question type (Phase 2); probabilities per word.", intents),
    stage(6, "Candidates and selection",
      "Each question type becomes a SQL candidate. Score = log P(type) + 6 × similarity(question, explanation) − 2 × penalties.",
      `<div class="table-scroll"><table class="data-table compact cand-table"><thead><tr><th>Type</th><th>SQL</th><th class="num">P(type)</th><th class="num">Similarity</th><th>Penalties</th><th class="num">Score</th></tr></thead><tbody>${cands}</tbody></table></div>` +
        (invalid.length ? `<p class="muted small">Rejected: ${invalid.slice(0, 6).map((c) => `${c.intent} (${escapeHtml(c.invalid)})`).join("; ")}</p>` : "")),
    stage(7, "PARSer-lite", "The chosen query split into clause fragments.", `<div class="frags">${frags}</div>`),
  ].join("");
}

async function ask(question) {
  const q = (question ?? $("q").value).trim();
  if (!q) return;
  $("q").value = q;
  $("ac").innerHTML = "";
  $("err").hidden = true;
  $("ask").disabled = true;
  $("ask").textContent = "Thinking…";
  try {
    const t = await api("/api/ask", { text: q });
    if (t.error) {
      $("err").textContent = t.error;
      $("err").hidden = false;
    }
    if (!t.sql) {
      $("answer").hidden = true;
      return;
    }
    $("a-lang").textContent = LANG_NAMES[t.lang];
    const res = t.result;
    $("a-count").textContent = res && res.rows ? `${res.rows.length}${res.truncated ? "+" : ""} row${res.rows.length === 1 ? "" : "s"}` : "";
    $("a-result").innerHTML = t.sql ? renderResult(res) : "";
    $("a-sql").innerHTML = t.sql ? highlightSql(t.sql) : "—";
    $("a-reason").innerHTML = t.reasoning.map((s) => `<li>${escapeHtml(s)}</li>`).join("");
    renderPipeline(t);
    $("answer").hidden = false;
  } catch (e) {
    $("err").textContent = e.message;
    $("err").hidden = false;
  } finally {
    $("ask").disabled = false;
    $("ask").textContent = "Ask";
  }
}

const suggest = debounce(async () => {
  const text = $("q").value;
  if (!text.trim()) {
    $("ac").innerHTML = "";
    return;
  }
  try {
    const d = await api(`/api/autocomplete?text=${encodeURIComponent(text)}`);
    $("ac").innerHTML = d.suggestions
      .filter((s) => s.word !== "</s>")
      .slice(0, 5)
      .map((s) => `<button class="suggestion" data-w="${escapeHtml(s.word)}">${escapeHtml(s.word)}</button>`)
      .join("");
    $("ac").querySelectorAll("[data-w]").forEach((b) =>
      b.addEventListener("click", () => {
        const t = $("q").value;
        const base = t.endsWith(" ") ? t : t.slice(0, t.lastIndexOf(" ") + 1);
        $("q").value = base + b.dataset.w + " ";
        $("q").focus();
        suggest();
      })
    );
  } catch (e) {
    $("ac").innerHTML = "";
  }
}, 150);

$("ask").addEventListener("click", () => ask());
$("q").addEventListener("keydown", (e) => e.key === "Enter" && ask());
$("q").addEventListener("input", suggest);
$("copy").addEventListener("click", () => navigator.clipboard && navigator.clipboard.writeText($("a-sql").textContent));
document.querySelectorAll("[data-q]").forEach((b) => b.addEventListener("click", () => ask(b.dataset.q)));
ask(document.querySelector("[data-q]").dataset.q);
