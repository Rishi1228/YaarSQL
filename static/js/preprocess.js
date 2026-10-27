// Preprocessing Lab: sends text to /api/preprocess and renders each step.

function chips(words, cls = "") {
  if (!words.length) return '<span class="muted">(empty)</span>';
  return words
    .map((w) => `<span class="tok ${cls}">${escapeHtml(w)}</span>`)
    .join("");
}

function step(title, desc, body) {
  return `<li class="step"><div class="step-head"><h3>${title}</h3><p>${desc}</p></div><div class="step-body">${body}</div></li>`;
}

function render(r) {
  $("s-lang").textContent = LANG_NAMES[r.language] || r.language;
  $("s-tokens").textContent = r.tokens.length;
  $("s-fixes").textContent = r.spelling_changes.length;
  $("s-stop").textContent = r.numbers.length - r.without_stopwords.length;

  const fixes = r.spelling_changes.length
    ? r.spelling_changes
        .map(
          (c) =>
            `<span class="fix"><s>${escapeHtml(c.from)}</s> → <b>${escapeHtml(c.to)}</b> <small>${c.rule}</small></span>`,
        )
        .join("")
    : `<span class="muted">${r.language === "hg" ? "No variants found." : "Only applied to Hinglish."}</span>`;

  const tokensWithScript = r.tokens
    .map(
      (t, i) =>
        `<span class="tok script-${r.scripts[i]}" title="${r.scripts[i]}">${escapeHtml(t)}</span>`,
    )
    .join("");

  // stems: show "word → stem" only where it changed
  const words = r.numbers;
  const stems = r.stems.map((s, i) =>
    s !== words[i] ? `${words[i]} → ${s}` : s,
  );

  $("steps").innerHTML = [
    step(
      "1. Unicode normalization",
      "NFC form, Devanagari digits → 0-9, nukta (ज़→ज) and chandrabindu (पाँच→पांच) folded, extra spaces removed.",
      `<div class="text-out">${escapeHtml(r.normalized)}</div>`,
    ),
    step(
      "2. Lowercasing",
      "Latin letters only; Devanagari has no upper/lower case.",
      `<div class="text-out">${escapeHtml(r.lowercased)}</div>`,
    ),
    step(
      "3. Sentence splitting",
      "Split on . ? ! and the Hindi full stop (।).",
      chips(r.sentences, "sent"),
    ),
    step(
      "4–5. Tokenization + script",
      "Regex tokenizer; colour shows the script of each token.",
      tokensWithScript,
    ),
    step(
      "6–7. Language identification",
      `Sentence detected as <b>${LANG_NAMES[r.language]}</b>. Each word is also tagged (see the token table).`,
      r.tokens
        .map(
          (t, i) =>
            `<span class="tok lang-${r.word_langs[i]}">${escapeHtml(t)}<sub>${r.word_langs[i]}</sub></span>`,
        )
        .join(""),
    ),
    step(
      "8. Hinglish spelling normalization",
      "Variants like kitney / ktne → kitne, by lexicon lookup or edit distance 1.",
      fixes,
    ),
    step(
      "9. Number words → digits",
      "five / paanch / पाँच → 5 (punctuation also dropped here).",
      chips(r.numbers),
    ),
    step(
      "10. Stopword removal",
      "Language-specific stopword list.",
      chips(r.without_stopwords),
    ),
    step(
      "11. Stemming",
      "Porter stemmer for English, suffix stripping for Hindi and Hinglish.",
      chips(stems),
    ),
    step(
      "Final tokens",
      "What the N-gram and HMM phases receive.",
      chips(r.final, "final"),
    ),
  ].join("");

  $("token-table").querySelector("tbody").innerHTML = r.tokens
    .map(
      (t, i) =>
        `<tr><td>${i + 1}</td><td class="tok-cell">${escapeHtml(t)}</td><td>${r.scripts[i]}</td><td><span class="tag lang-${r.word_langs[i]}">${r.word_langs[i]}</span></td></tr>`,
    )
    .join("");

  $("result").hidden = false;
}

async function run() {
  $("error").hidden = true;
  const res = await fetch("/api/preprocess", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: $("text").value,
      lang: $("lang").value,
      drop_stopwords: $("stop").checked,
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    $("error").textContent = data.error || "Something went wrong.";
    $("error").hidden = false;
    return;
  }
  render(data);
}

$("run").addEventListener("click", run);
$("text").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
});
document.querySelectorAll("[data-example]").forEach((b) =>
  b.addEventListener("click", () => {
    $("text").value = b.dataset.example;
    run();
  }),
);
run();
