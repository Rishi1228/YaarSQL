// Small helpers shared by every page.
const $ = (id) => document.getElementById(id);

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function api(url, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(url, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function fmt(x, digits = 3) {
  if (x === null || x === undefined) return "∞";
  if (x === 0) return "0";
  if (Math.abs(x) < 0.001) return x.toExponential(2);
  return Number(x).toFixed(digits);
}

function pct(x) {
  return `${(100 * x).toFixed(1)}%`;
}

const LANG_NAMES = { en: "English", hi: "Hindi", hg: "Hinglish" };

// Read colours from CSS variables so charts match the page.
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function debounce(fn, ms = 200) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}
