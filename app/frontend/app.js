// Practica Vucetich - vanilla JS
// Estado:
//   currentAttempt: { id, source } | null
//   answeredFor: id del intento ya respondido (bloquea re-clicks)

const API = {
  sample: () => fetch("/api/sample").then(handleJson),
  answer: (id, k) => fetch("/api/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, klass_answered: k }),
  }).then(handleJson),
  stats: () => fetch("/api/stats").then(handleJson),
};

const VALID_KLASS = ["A", "I", "E", "V"];
const KLASS_NAME = {
  A: "Arco",
  I: "Presilla Interna",
  E: "Presilla Externa",
  V: "Verticilo",
};

const $ = (id) => document.getElementById(id);
const els = {
  img: $("fingerprint"),
  badge: $("source-badge"),
  feedback: $("feedback"),
  feedbackIcon: $("feedback-icon"),
  feedbackText: $("feedback-text"),
  nextBtn: $("next-btn"),
  total: $("stat-total"),
  correct: $("stat-correct"),
  accuracy: $("stat-accuracy"),
  streak: $("stat-streak"),
  best: $("stat-best"),
  perClassBody: $("per-class").querySelector("tbody"),
  answers: Array.from(document.querySelectorAll(".answer-btn")),
};

let currentAttempt = null;
let answeredFor = null;

async function handleJson(resp) {
  if (!resp.ok) {
    const text = await resp.text().catch(() => resp.statusText);
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }
  return resp.json();
}

function clearFeedbackStyling() {
  els.feedback.hidden = true;
  els.feedback.classList.remove("correct", "incorrect");
  els.answers.forEach((b) => {
    b.classList.remove("was-correct", "was-incorrect", "was-correct-target");
    b.disabled = false;
  });
}

async function loadNextSample() {
  clearFeedbackStyling();
  els.img.alt = "cargando huella...";
  els.badge.hidden = true;

  try {
    const data = await API.sample();
    currentAttempt = { id: data.id, source: data.source };
    answeredFor = null;
    els.img.src = `data:image/png;base64,${data.png_b64}`;
    els.img.alt = `huella #${data.id}`;
    els.badge.hidden = false;
    els.badge.textContent = data.source === "real" ? "REAL" : "GAN";
  } catch (err) {
    els.img.alt = "error: " + err.message;
    console.error(err);
  }
}

async function submitAnswer(klass) {
  if (!currentAttempt) return;
  if (answeredFor === currentAttempt.id) return; // bloqueo idempotente

  const attemptId = currentAttempt.id;
  answeredFor = attemptId;

  // disable all buttons mientras va el POST
  els.answers.forEach((b) => (b.disabled = true));

  let result;
  try {
    result = await API.answer(attemptId, klass);
  } catch (err) {
    console.error(err);
    answeredFor = null;
    els.answers.forEach((b) => (b.disabled = false));
    alert("Error al mandar la respuesta: " + err.message);
    return;
  }

  // marcar boton elegido + (si fue mal) tambien resaltar el correcto
  els.answers.forEach((b) => {
    if (b.dataset.klass === klass) {
      b.classList.add(result.correct ? "was-correct" : "was-incorrect");
    }
    if (!result.correct && b.dataset.klass === result.klass_asked) {
      b.classList.add("was-correct-target");
    }
  });

  els.feedback.hidden = false;
  els.feedback.classList.add(result.correct ? "correct" : "incorrect");
  els.feedbackIcon.textContent = result.correct ? "+" : "x";
  if (result.correct) {
    els.feedbackText.textContent =
      `Correcto. Era ${KLASS_NAME[result.klass_asked]} (${result.klass_asked}).`;
  } else {
    els.feedbackText.textContent =
      `Incorrecto. Era ${KLASS_NAME[result.klass_asked]} (${result.klass_asked}), no ${KLASS_NAME[klass]} (${klass}).`;
  }
  renderStatsFromAnswer(result);
  els.nextBtn.focus();
}

function renderStatsFromAnswer(result) {
  // El POST /api/answer devuelve agregados sufientes para no pegarle a /stats
  // por cada respuesta. Los breakdowns por clase si requieren /stats.
  els.total.textContent = result.total;
  els.accuracy.textContent = (result.accuracy * 100).toFixed(1) + " %";
  els.streak.textContent = result.current_streak;
  els.best.textContent = result.best_streak;
  els.correct.textContent = Math.round(result.accuracy * result.total);
  // El breakdown por clase lo refrescamos en background
  refreshFullStats();
}

async function refreshFullStats() {
  try {
    const s = await API.stats();
    els.total.textContent = s.total;
    els.correct.textContent = s.correct;
    els.accuracy.textContent = s.total
      ? (s.accuracy * 100).toFixed(1) + " %"
      : "—";
    els.streak.textContent = s.current_streak;
    els.best.textContent = s.best_streak;

    // tabla por clase
    els.perClassBody.innerHTML = "";
    for (const sym of VALID_KLASS) {
      const row = document.createElement("tr");
      const bucket = s.per_class[sym] || { total: 0, correct: 0 };
      const pct = bucket.total
        ? ((bucket.correct / bucket.total) * 100).toFixed(0)
        : "—";
      row.innerHTML = `
        <td>${sym}</td>
        <td>${bucket.correct}</td>
        <td>${bucket.total}</td>
        <td>${pct}${bucket.total ? " %" : ""}</td>`;
      els.perClassBody.appendChild(row);
    }
  } catch (err) {
    console.error("stats error", err);
  }
}

// --- wiring ---

els.answers.forEach((btn) => {
  btn.addEventListener("click", () => submitAnswer(btn.dataset.klass));
});
els.nextBtn.addEventListener("click", loadNextSample);

document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement) return;
  const k = e.key.toUpperCase();
  if (VALID_KLASS.includes(k) && answeredFor !== currentAttempt?.id) {
    submitAnswer(k);
    e.preventDefault();
    return;
  }
  if ((e.key === "Enter" || e.key === " ") && answeredFor === currentAttempt?.id) {
    loadNextSample();
    e.preventDefault();
  }
});

// boot
refreshFullStats();
loadNextSample();
