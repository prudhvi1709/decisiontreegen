import { render, html } from "https://cdn.jsdelivr.net/npm/lit-html@3/+esm";
import { unsafeHTML } from "https://cdn.jsdelivr.net/npm/lit-html@3/directives/unsafe-html.js";
import { Marked } from "https://cdn.jsdelivr.net/npm/marked@13/+esm";

const marked = new Marked();

export function showLoading(show) {
  document.getElementById("loadingOverlay").classList.toggle("d-none", !show);
}

export function showError(message) {
  const alertHtml = `
    <div class="alert alert-danger alert-dismissible fade show" role="alert">
      <i class="bi bi-exclamation-triangle me-2"></i>
      ${message}
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    </div>
  `;
  document.body.insertAdjacentHTML("afterbegin", alertHtml);
}

export function markdownToHtml(markdown) {
  return markdown ? marked.parse(markdown) : "";
}

export function renderTreeCollapsible(node, branchLabel = null) {
  if (node.prediction !== undefined) {
    // Render YES/NO badge and prediction in a flex row if branchLabel is provided
    if (branchLabel) {
      return `<div style="display: flex; align-items: center; gap: 0.5em; margin-bottom: 0.25em;">
        <span class="badge bg-${branchLabel === 'YES' ? 'success' : 'danger'} me-2">${branchLabel}</span>
        <span class="alert alert-success border-success text-break mb-0 py-1 px-2 d-inline-flex align-items-center" style="font-size: 1em;">
          <i class="bi bi-check-circle-fill me-2"></i>
          <strong>Prediction:</strong> <span class="badge bg-success ms-2">${node.prediction}</span>
        </span>
      </div>`;
    } else {
      // Root node prediction (shouldn't happen, but fallback)
      return `<span class="alert alert-success border-success text-break">
        <i class="bi bi-check-circle-fill me-2"></i>
        <strong>Prediction:</strong> <span class="badge bg-success ms-2">${node.prediction}</span>
      </span>`;
    }
  }

  const threshold = typeof node.threshold === "number" ? node.threshold.toFixed(3) : node.threshold;

  return `
    <details class="mb-2">
      <summary class="text-break fw-bold text-primary cursor-pointer" style="cursor: pointer;">
        <i class="bi bi-diagram-3 me-2"></i>
        Is <strong>${node.feature}</strong> &lt; <strong>${threshold}</strong>?
      </summary>
      <div class="ms-4 mt-2">
        ${renderTreeCollapsible(node.left, 'YES')}
        ${renderTreeCollapsible(node.right, 'NO')}
      </div>
    </details>
  `;
}

export function setupFormPersistence() {
  const inputs = document.querySelectorAll('input:not([type="file"]), select, textarea');
  
  inputs.forEach(input => {
    const savedValue = localStorage.getItem(`form_${input.id}`);
    if (savedValue && input.type !== 'file') {
      input.value = savedValue;
    }
    
    input.addEventListener('input', () => {
      localStorage.setItem(`form_${input.id}`, input.value);
    });
  });
}

export function displayMetrics(metrics) {
  const container = document.getElementById("metricsGrid");
  const metricsData = [
    { label: "Accuracy", value: (metrics.accuracy || 0).toFixed(3) },
    { label: "Precision", value: (metrics.precision || 0).toFixed(3) },
    { label: "Recall", value: (metrics.recall || 0).toFixed(3) },
    { label: "F1 Score", value: (metrics.f1_score || 0).toFixed(3) },
    { label: "False Positives", value: metrics.fp || 0 },
    { label: "False Negatives", value: metrics.fn || 0 },
  ];

  container.innerHTML = metricsData
    .map(metric => `
    <div class="col-lg-2 col-md-4 col-6">
      <div class="bg-light p-3 rounded text-center">
        <div class="fs-4 fw-bold text-primary">${metric.value}</div>
        <div class="small text-muted">${metric.label}</div>
      </div>
    </div>`)
    .join("");
}

export function displayDecisionTree(tree) {
  const container = document.getElementById("decisionTree");
  
  const buttonHtml = `
    <div class="mb-3">
      <button id="expandAllBtn" class="btn btn-outline-primary btn-sm">
        <i class="bi bi-arrows-expand me-1"></i>
        Expand All
      </button>
    </div>
  `;
  
  container.innerHTML = buttonHtml + renderTreeCollapsible(tree);
  
  const expandAllBtn = document.getElementById("expandAllBtn");
  let isExpanded = false;
  
  expandAllBtn.addEventListener("click", () => {
    const details = container.querySelectorAll("details");
    isExpanded = !isExpanded;
    
    details.forEach(detail => detail.open = isExpanded);
    
    expandAllBtn.innerHTML = isExpanded ? 
      '<i class="bi bi-arrows-collapse me-1"></i>Collapse All' :
      '<i class="bi bi-arrows-expand me-1"></i>Expand All';
  });
}

export function renderMarkdownResponse(response, container) {
  const rendered = html`${unsafeHTML(marked.parse(response))}`;
  render(rendered, container);
}