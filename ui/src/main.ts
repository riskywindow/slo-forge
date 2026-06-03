import "./styles.css";
import { renderDashboard } from "./components";
import { renderFabricDashboard } from "./fabric-components";
import {
  FabricArtifactValidationError,
  fetchArtifactDocument,
  parseArtifactDocument,
} from "./fabric-parser";
import {
  ArtifactValidationError,
} from "./parser";
import type { ArtifactDocument } from "./fabric-types";

const appNode = document.querySelector<HTMLDivElement>("#app");
if (appNode === null) throw new Error("Missing #app mount element");
const app = appNode;

const environmentSource = import.meta.env.VITE_REPORT_URL;
const initialSource =
  new URLSearchParams(window.location.search).get("data") ??
  environmentSource ??
  "./report-data.json";

function shell(content: string, source: string, kind: ArtifactDocument["kind"] = "logical"): string {
  const navigation = kind === "fabric"
    ? '<a href="#fabric-topology">Topology</a><a href="#fabric-placement">Placement</a><a href="#fabric-communication">Communication</a><a href="#fabric-autopsy">Autopsy</a><a href="#fabric-recovery">Recovery</a>'
    : '<a href="#plan">Plan</a><a href="#evidence">Evidence</a><a href="#frontier">Frontier</a><a href="#runtime">Runtime</a><a href="#faults">Faults</a>';
  return `<header class="app-bar">
    <a class="brand" href="#dashboard" aria-label="SLOForge artifact explorer home"><span class="brand-mark">S</span><span><strong>SLOForge</strong><small>${kind === "fabric" ? "fabric explorer" : "artifact explorer"}</small></span></a>
    <nav aria-label="Evidence sections">${navigation}</nav>
    <details class="source-control"><summary>Load artifact</summary><form id="source-form"><label for="source-url">Artifact URL</label><div><input id="source-url" name="source" type="text" value="${escapeAttribute(source)}" spellcheck="false" autocomplete="off"/><button type="submit">Load</button></div><label class="file-label" for="artifact-file">or choose local JSON<input id="artifact-file" type="file" accept="application/json,.json"/></label><p>Tip: append <code>?data=/path/report-data.json</code> to configure a static deployment.</p></form></details>
  </header>${content}<footer><p>Rendered locally from validated SLOForge evidence. No data leaves this browser.</p></footer>`;
}

function escapeAttribute(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function bindShell(source: string): void {
  const form = document.querySelector<HTMLFormElement>("#source-form");
  const fileInput = document.querySelector<HTMLInputElement>("#artifact-file");
  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    const sourceValue = formData.get("source");
    const nextSource = typeof sourceValue === "string" ? sourceValue.trim() : "";
    if (nextSource.length === 0) return;
    const url = new URL(window.location.href);
    url.searchParams.set("data", nextSource);
    window.history.replaceState({}, "", url);
    void loadFromUrl(nextSource);
  });
  fileInput?.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file !== undefined) void loadFromFile(file, source);
  });
}

function renderDocument(document: ArtifactDocument, source: string): void {
  const dashboard = document.kind === "fabric"
    ? renderFabricDashboard(document.value)
    : renderDashboard(document.value);
  app.innerHTML = shell(dashboard, source, document.kind);
  bindShell(source);
}

function renderLoading(source: string): void {
  app.innerHTML = shell(
    `<main id="dashboard"><section class="load-state" aria-busy="true"><div class="loader"></div><h1>Loading evidence</h1><p>${escapeAttribute(source)}</p></section></main>`,
    source,
  );
  bindShell(source);
}

function renderError(error: unknown, source: string): void {
  const message = error instanceof Error ? error.message : String(error);
  const details =
    error instanceof ArtifactValidationError || error instanceof FabricArtifactValidationError
      ? `<ul>${error.problems.map((problem) => `<li>${escapeAttribute(problem)}</li>`).join("")}</ul>`
      : "";
  app.innerHTML = shell(
    `<main id="dashboard"><section class="load-state error-state" role="alert"><span class="error-icon">!</span><p class="eyebrow">Artifact unavailable</p><h1>Evidence could not be rendered</h1><p>${escapeAttribute(message)}</p>${details}<p>Serve a real <code>report-data.json</code>, enter its URL above, or choose the file locally.</p></section></main>`,
    source,
  );
  bindShell(source);
}

async function loadFromUrl(source: string): Promise<void> {
  renderLoading(source);
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 10_000);
  try {
    renderDocument(await fetchArtifactDocument(source, controller.signal), source);
  } catch (error: unknown) {
    renderError(error, source);
  } finally {
    window.clearTimeout(timeout);
  }
}

async function loadFromFile(file: File, displayedSource: string): Promise<void> {
  const maximumBytes = 50 * 1024 * 1024;
  if (file.size > maximumBytes) {
    renderError(new Error("Artifact exceeds the 50 MiB local-file limit"), displayedSource);
    return;
  }
  renderLoading(file.name);
  try {
    renderDocument(parseArtifactDocument(JSON.parse(await file.text()) as unknown), file.name);
  } catch (error: unknown) {
    renderError(error, file.name);
  }
}

void loadFromUrl(initialSource);
