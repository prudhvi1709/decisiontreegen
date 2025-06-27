import { showError, showSuccess, showInfo, showLoading } from './utils.js';

export function parseCSV(text) {
  const lines = text.trim().split("\n");
  const firstLine = lines[0];
  
  const delimiter = firstLine.includes(";") && firstLine.split(";").length > firstLine.split(",").length ? ";" : ",";
  
  const headers = firstLine.split(delimiter).map(h => h.trim().replace(/"/g, ""));
  const rows = lines.slice(1).map(line => {
    const values = line.split(delimiter).map(v => v.trim().replace(/"/g, ""));
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] || "";
    });
    return row;
  });
  return { headers, rows };
}

export async function parseXLSX(file) {
  const arrayBuffer = await file.arrayBuffer();
  const { read, utils } = await import("https://cdn.skypack.dev/xlsx@0.18.5");
  const workbook = read(arrayBuffer);
  const firstSheetName = workbook.SheetNames[0];
  const worksheet = workbook.Sheets[firstSheetName];
  const jsonData = utils.sheet_to_json(worksheet, { header: 1 });

  const headers = jsonData[0];
  const rows = jsonData.slice(1).map(row => {
    const rowObj = {};
    headers.forEach((header, index) => {
      rowObj[header] = row[index] || "";
    });
    return rowObj;
  });

  return { headers, rows };
}

export async function handleFileUpload(event, processDataCallback) {
  const file = event.target.files[0];
  if (!file) return;

  try {
    showLoading(true);
    let data;

    if (file.name.endsWith(".csv")) {
      const text = await file.text();
      data = parseCSV(text);
    } else if (file.name.endsWith(".xlsx") || file.name.endsWith(".xls")) {
      data = await parseXLSX(file);
    } else {
      throw new Error("Unsupported file format. Please use CSV or XLSX files.");
    }

    showSuccess(`File uploaded successfully! Found ${data.rows.length} rows with ${data.headers.length} columns.`);
    processDataCallback(data);
  } catch (error) {
    showError("Failed to process file: " + error.message);
  } finally {
    showLoading(false);
  }
}

export async function loadSampleDataset(url, target, processDataCallback) {
  try {
    showLoading(true);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const csvText = await response.text();
    const data = parseCSV(csvText);
    showSuccess(`Sample dataset loaded successfully! Found ${data.rows.length} rows with ${data.headers.length} columns.`);
    processDataCallback(data, target);
  } catch (error) {
    showError("Failed to load sample dataset: " + error.message);
  } finally {
    showLoading(false);
  }
}

export function displayDataPreview(data) {
  const preview = document.getElementById("dataPreview");
  const table = document.getElementById("previewTable");
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");

  thead.innerHTML = `<tr>${data.headers.map(h => `<th>${h}</th>`).join("")}</tr>`;

  const previewRows = data.rows.slice(0, 20);
  tbody.innerHTML = previewRows
    .map(row => `<tr>${data.headers.map(h => `<td>${row[h] || ""}</td>`).join("")}</tr>`)
    .join("");

  preview.classList.remove("d-none");
}

export function populateTargetColumn(headers, defaultTarget = null) {
  const select = document.getElementById("targetColumn");
  select.innerHTML = headers
    .map(h => `<option value="${h}"${h === defaultTarget ? " selected" : ""}>${h}</option>`)
    .join("");
}