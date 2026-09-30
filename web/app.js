// isso-rag front end: two views (Ask, How it works) and the chat transcript.
//
// The API is the Cloud Run service. It is called directly from the browser
// (CORS) rather than through a proxy, so the per-IP rate limit sees each
// visitor's own address.

// Served from localhost, the page talks to a local `uvicorn main:app`.
const LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
const API_URL = LOCAL
  ? "http://localhost:8000/chat"
  : "https://isso-rag-chatbot-215864209187.us-central1.run.app/chat";
const MAX_CHARS = 1000;
// A cold start plus the Gemini calls has been measured at over a minute, so
// the timeout is generous and the waiting copy says so after a while.
const TIMEOUT_MS = 90_000;
const SLOW_AFTER_MS = 10_000;
const SPINNER_DELAY_MS = 150;

const $ = (selector) => document.querySelector(selector);

const thread = $("#thread");
const starters = $("#starters");
const composer = $("#composer");
const input = $("#question");
const send = $("#send");
const help = $("#question-help");
const count = $("#count");
const countValue = $("#count-value");
const entryTemplate = $("#entry-template");

const HELP_DEFAULT = help.textContent;
let busy = false;

// The composer is sticky, so anything scrolled into view must stop above it.
new ResizeObserver(([entry]) => {
  document.documentElement.style.setProperty(
    "--composer-height",
    `${Math.ceil(entry.borderBoxSize[0].blockSize)}px`,
  );
}).observe(composer);

/* ---------- Views ---------- */

function showView() {
  const name = location.hash === "#how" ? "how" : "ask";
  for (const view of document.querySelectorAll("[data-view]")) {
    view.hidden = view.dataset.view !== name;
  }
  for (const link of document.querySelectorAll("[data-view-link]")) {
    if (link.dataset.viewLink === name) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  document.title = name === "how"
    ? "How it works · isso-rag"
    : "isso-rag · Answers from ISSO pages for incoming students";
}

window.addEventListener("hashchange", () => {
  showView();
  window.scrollTo({ top: 0 });
});
showView();

/* ---------- Composer ---------- */

function questionLength() {
  return input.value.trim().length;
}

function syncComposer() {
  const length = questionLength();
  const over = length > MAX_CHARS;
  countValue.textContent = length.toLocaleString("en-US");
  count.dataset.over = String(over);
  input.setAttribute("aria-invalid", String(over));
  if (over) {
    setHelp(`That’s ${(length - MAX_CHARS).toLocaleString("en-US")} characters over the limit. Shorten it to send.`, "error");
  } else if (help.dataset.tone === "error") {
    setHelp(HELP_DEFAULT);
  }
  send.disabled = busy || length === 0 || over;
  if (!busy) send.dataset.state = "idle";
}

function setHelp(text, tone) {
  help.textContent = text;
  if (tone) help.dataset.tone = tone;
  else delete help.dataset.tone;
}

// Grow the textarea with its content, up to the CSS max-height.
function autosize() {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight + 2}px`;
}

input.addEventListener("input", () => {
  autosize();
  syncComposer();
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (busy || !question || question.length > MAX_CHARS) return;
  input.value = "";
  autosize();
  ask(question);
});

for (const button of document.querySelectorAll(".starter")) {
  button.addEventListener("click", () => ask(button.textContent.trim()));
}

/* ---------- Asking ---------- */

async function ask(question, existingEntry) {
  busy = true;
  setStartersDisabled(true);
  send.disabled = true;
  send.dataset.state = "loading";

  const entry = existingEntry ?? addEntry(question);
  const answerSlot = entry.querySelector(".entry__a");
  const pending = renderPending(answerSlot);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  let outcome;
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question }),
      signal: controller.signal,
    });
    const body = await response.json().catch(() => null);
    outcome = interpret(response.status, body);
  } catch (error) {
    outcome = error.name === "AbortError"
      ? { problem: "The answer took more than 90 seconds, so this page stopped waiting. Try again." }
      : { problem: "This page couldn’t reach the assistant. Check your connection, then try again." };
  } finally {
    clearTimeout(timeout);
    pending.stop();
  }

  if (outcome.answer) {
    renderAnswer(answerSlot, outcome.answer);
    // Start reading at the question; long answers continue below it.
    entry.scrollIntoView({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" });
    send.dataset.state = "idle";
  } else {
    renderProblem(answerSlot, outcome.problem, outcome.retry !== false ? () => retry(entry, question) : null);
    send.dataset.state = "error";
  }

  busy = false;
  setStartersDisabled(false);
  syncComposer();
  if (send.dataset.state === "idle") input.focus({ preventScroll: true });
}

function retry(entry, question) {
  if (busy) return;
  ask(question, entry);
}

function interpret(status, body) {
  if (status === 200 && body && typeof body.response === "string") {
    return { answer: body.response };
  }
  if (status === 429) {
    const message = body && typeof body.message === "string"
      ? body.message
      : "Too many questions in a short time.";
    return { problem: `${message} Each visitor can ask 5 questions a minute and 50 a day.` };
  }
  if (status === 422) {
    return { problem: "The assistant couldn’t read that question. Keep it under 1,000 characters and try again.", retry: false };
  }
  return { problem: `The assistant returned an error (${status}). Try again in a moment.` };
}

/* ---------- Rendering ---------- */

function addEntry(question) {
  const entry = entryTemplate.content.firstElementChild.cloneNode(true);
  entry.querySelector(".entry__q").textContent = question;
  thread.append(entry);
  starters.hidden = true;
  entry.scrollIntoView({ block: "nearest", behavior: prefersReducedMotion() ? "auto" : "smooth" });
  return entry;
}

// The spinner waits SPINNER_DELAY_MS before showing, so fast answers never
// flash it, and the copy turns honest about cold starts after SLOW_AFTER_MS.
function renderPending(slot) {
  slot.replaceChildren();
  const row = document.createElement("p");
  row.className = "pending";
  row.dataset.visible = "false";
  row.setAttribute("role", "status");
  const spinner = document.createElement("span");
  spinner.className = "spinner";
  spinner.setAttribute("aria-hidden", "true");
  const text = document.createElement("span");
  text.textContent = "Reading the ISSO pages…";
  row.append(spinner, text);
  slot.append(row);

  const showTimer = setTimeout(() => { row.dataset.visible = "true"; }, SPINNER_DELAY_MS);
  const slowTimer = setTimeout(() => {
    text.textContent = "Still working. After a quiet spell the first answer can take up to a minute.";
  }, SLOW_AFTER_MS);

  return {
    stop() {
      clearTimeout(showTimer);
      clearTimeout(slowTimer);
      row.remove();
    },
  };
}

// The API returns plain text that may end with one or more lines of the form
// "More info: <Heading> (<URL>)". Everything is inserted as text, never HTML.
function renderAnswer(slot, text) {
  const marker = text.search(/\n?\s*More info:/);
  const body = (marker === -1 ? text : text.slice(0, marker)).trim();
  const tail = marker === -1 ? "" : text.slice(marker);

  const answer = document.createElement("div");
  answer.className = "answer";
  for (const block of body.split(/\n{2,}/)) {
    if (!block.trim()) continue;
    const p = document.createElement("p");
    p.textContent = block.trim();
    answer.append(p);
  }

  const citations = [...tail.matchAll(/More info:\s*(.+?)\s*\((https:\/\/[^\s)]+)\)/g)];
  if (citations.length) {
    for (const [, heading, url] of citations) {
      answer.append(renderSource(heading, url));
    }
  } else if (tail.trim()) {
    const p = document.createElement("p");
    p.textContent = tail.trim();
    answer.append(p);
  }

  slot.replaceChildren(answer);
}

function renderSource(heading, url) {
  const source = document.createElement("p");
  source.className = "source";

  const label = document.createElement("span");
  label.className = "source__label";
  label.textContent = "Source";

  const link = document.createElement("a");
  link.className = "source__link";
  link.href = url;
  link.rel = "noopener";
  link.target = "_blank";
  // "Entering the U.S. > 2 What documents…": drop the FAQ's own numbering.
  link.textContent = heading.replace(/\s>\s(\d+\s+)?/g, " › ");

  const host = document.createElement("span");
  host.className = "source__host";
  try {
    host.textContent = new URL(url).hostname;
  } catch {
    host.textContent = url;
  }

  source.append(label, link, host);
  return source;
}

function renderProblem(slot, message, onRetry) {
  const box = document.createElement("div");
  box.className = "problem";
  box.setAttribute("role", "alert");

  const glyph = document.createElement("span");
  glyph.className = "problem__glyph";
  glyph.setAttribute("aria-hidden", "true");
  glyph.textContent = "!";

  const text = document.createElement("p");
  text.textContent = message;

  box.append(glyph, text);
  if (onRetry) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "problem__retry";
    button.textContent = "Ask again";
    button.addEventListener("click", onRetry);
    box.append(button);
  }
  slot.replaceChildren(box);
}

function setStartersDisabled(disabled) {
  for (const button of document.querySelectorAll(".starter")) button.disabled = disabled;
}

function prefersReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

syncComposer();
