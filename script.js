import { asyncLLM } from "https://cdn.jsdelivr.net/npm/asyncllm@2";
import { render, html } from "https://cdn.jsdelivr.net/npm/lit-html@3/+esm";
import { unsafeHTML } from "https://cdn.jsdelivr.net/npm/lit-html@3/directives/unsafe-html.js";
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.mjs";
import { Marked } from "https://cdn.jsdelivr.net/npm/marked@13/+esm";

// Global state
let currentData = null;
let currentColumns = [];
let currentCode = "";
let currentDecisionTree = null;
let pyodide = null;
const marked = new Marked();

// Configuration will be loaded from config.json
let config = null;

// Load configuration
async function loadConfig() {
  const response = await fetch("config.json");
  config = await response.json();
  document.getElementById("baseUrlInput").value = config.defaultSettings.baseUrl;
  document.getElementById("modelInput").value = config.defaultSettings.model;
  document.getElementById("maxDepth").value = config.defaultSettings.maxDepth;
}

// Initialize the app
document.addEventListener("DOMContentLoaded", async () => {
  await loadConfig();
  initializePyodide();
  setupEventListeners();
  renderSampleDatasets();
  setupFormPersistence();
});

// Initialize Pyodide
async function initializePyodide() {
  try {
    showLoading(true);
    pyodide = await loadPyodide();
    await pyodide.loadPackage(["pandas", "scikit-learn", "numpy"]);
    
    // Import common packages globally
    await pyodide.runPython(`
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    `);
  } catch (error) {
    showError("Failed to initialize Python environment: " + error.message);
  } finally {
    showLoading(false);
  }
}

// Setup form persistence
function setupFormPersistence() {
  // Remove saveform due to compatibility issues
  // Implement simple localStorage persistence instead
  const inputs = document.querySelectorAll('input:not([type="file"]), select, textarea');
  
  inputs.forEach(input => {
    // Load saved value
    const savedValue = localStorage.getItem(`form_${input.id}`);
    if (savedValue && input.type !== 'file') {
      input.value = savedValue;
    }
    
    // Save on change
    input.addEventListener('input', () => {
      localStorage.setItem(`form_${input.id}`, input.value);
    });
  });
}

// Setup event listeners
function setupEventListeners() {
  document.getElementById("fileInput").addEventListener("change", handleFileUpload);
  document.getElementById("targetColumn").addEventListener("change", updatePrompt);
  document.getElementById("maxDepth").addEventListener("change", updatePrompt);
  document.getElementById("analyzeBtn").addEventListener("click", (e) => {
    e.preventDefault();
    analyzeData().catch(error => {
      showError("Analysis failed: " + error.message);
    });
  });
  document.getElementById("reviseBtn").addEventListener("click", reviseAnalysis);
}

// Render sample datasets
function renderSampleDatasets() {
  const container = document.getElementById("sampleDatasets");
  if (!config || !config.sampleDatasets) return;
  
  const cardsHtml = config.sampleDatasets
    .map(
      (dataset) => `
    <div class="col-md-6 col-lg-3 mb-3">
      <div class="card h-100 cursor-pointer" onclick="loadSampleDataset('${dataset.url}', '${dataset.target}')" style="cursor: pointer; transition: transform 0.2s;" onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
        <div class="card-body text-center">
          <i class="${dataset.icon} display-4 text-primary mb-3"></i>
          <h5 class="card-title">${dataset.title}</h5>
          <p class="card-text">${dataset.description}</p>
          <small class="text-muted">Target: ${dataset.target}</small>
        </div>
      </div>
    </div>
  `,
    )
    .join("");
  container.innerHTML = cardsHtml;
}

// Load sample dataset
window.loadSampleDataset = async (url, target) => {
  try {
    showLoading(true);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const csvText = await response.text();
    const data = parseCSV(csvText);
    processData(data, target);
  } catch (error) {
    showError("Failed to load sample dataset: " + error.message);
  } finally {
    showLoading(false);
  }
};

// Handle file upload
async function handleFileUpload(event) {
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

    processData(data);
  } catch (error) {
    showError("Failed to process file: " + error.message);
  } finally {
    showLoading(false);
  }
}

// Parse CSV
function parseCSV(text) {
  const lines = text.trim().split("\n");
  const firstLine = lines[0];
  
  // Detect delimiter (comma or semicolon)
  const delimiter = firstLine.includes(";") && firstLine.split(";").length > firstLine.split(",").length ? ";" : ",";
  
  const headers = firstLine.split(delimiter).map((h) => h.trim().replace(/"/g, ""));
  const rows = lines.slice(1).map((line) => {
    const values = line.split(delimiter).map((v) => v.trim().replace(/"/g, ""));
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] || "";
    });
    return row;
  });
  return { headers, rows };
}

// Parse XLSX
async function parseXLSX(file) {
  const arrayBuffer = await file.arrayBuffer();
  const { read, utils } = await import("https://cdn.skypack.dev/xlsx@0.18.5");
  const workbook = read(arrayBuffer);
  const firstSheetName = workbook.SheetNames[0];
  const worksheet = workbook.Sheets[firstSheetName];
  const jsonData = utils.sheet_to_json(worksheet, { header: 1 });

  const headers = jsonData[0];
  const rows = jsonData.slice(1).map((row) => {
    const rowObj = {};
    headers.forEach((header, index) => {
      rowObj[header] = row[index] || "";
    });
    return rowObj;
  });

  return { headers, rows };
}

// Process data after loading
function processData(data, defaultTarget = null) {
  currentData = data;
  currentColumns = data.headers;

  // Show data preview
  displayDataPreview(data);

  // Populate target column dropdown
  populateTargetColumn(data.headers, defaultTarget);

  // Update prompt
  updatePrompt();

  // Show step 2
  document.getElementById("step2").classList.remove("d-none");
  document.getElementById("step2").scrollIntoView({ behavior: "smooth" });
}

// Display data preview
function displayDataPreview(data) {
  const preview = document.getElementById("dataPreview");
  const table = document.getElementById("previewTable");
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");

  // Create header
  thead.innerHTML = `<tr>${data.headers.map((h) => `<th>${h}</th>`).join("")}</tr>`;

  // Create rows (first 20)
  const previewRows = data.rows.slice(0, 20);
  tbody.innerHTML = previewRows
    .map((row) => `<tr>${data.headers.map((h) => `<td>${row[h] || ""}</td>`).join("")}</tr>`)
    .join("");

  preview.classList.remove("d-none");
}

// Populate target column dropdown
function populateTargetColumn(headers, defaultTarget = null) {
  const select = document.getElementById("targetColumn");
  select.innerHTML = headers
    .map((h) => `<option value="${h}"${h === defaultTarget ? " selected" : ""}>${h}</option>`)
    .join("");
}

// Update prompt based on selections
function updatePrompt() {
  const target = document.getElementById("targetColumn").value;
  const depth = document.getElementById("maxDepth").value;
  const prompt = `Predict ${target} using a decision tree. Max depth: ${depth}.`;
  document.getElementById("promptText").value = prompt;
}

// Analyze data with LLM
async function analyzeData() {
  const apiKey = document.getElementById("apiKeyInput").value;
  const baseUrl = document.getElementById("baseUrlInput").value;
  const model = document.getElementById("modelInput").value;
  const prompt = document.getElementById("promptText").value;

  if (!apiKey) {
    showError("Please enter your OpenAI API key in the advanced settings.");
    return;
  }

  if (!currentData) {
    showError("Please upload a dataset first.");
    return;
  }

  try {
    showLoading(true);
    await streamLLMResponse(baseUrl, apiKey, model, prompt);
  } catch (error) {
    showError("Analysis failed: " + error.message);
  } finally {
    showLoading(false);
  }
}

// Stream LLM response
async function streamLLMResponse(baseUrl, apiKey, model, userPrompt) {
  const systemPrompt = `You are a data science expert. Generate Python code to create a decision tree model for the given dataset.

Dataset columns: ${currentColumns ? currentColumns.join(", ") : "Unknown"}
Data sample: ${currentData && currentData.rows ? JSON.stringify(currentData.rows.slice(0, 3)) : "No data"}

Your task is to:
1. Prepare the data for training
2. Create and train a decision tree model

The DataFrame 'df' is already created and available. Your code should:
1. Handle any missing values in the data
2. Select appropriate features and target columns
3. Create and train a decision tree model named 'model'

Notes:
- All necessary imports are already done (pandas, numpy, sklearn)
- Use DecisionTreeRegressor for numeric targets
- Use DecisionTreeClassifier for categorical targets
- Set appropriate hyperparameters (e.g., max_depth)

Example:
\`\`\`python
# Handle missing values
df.fillna(df.median(), inplace=True)

# Select target and features
target_col = 'price'  # choose appropriate target
feature_cols = [col for col in df.columns if col != target_col]

# Prepare data
X = df[feature_cols]
y = df[target_col]

# Create and train model
model = DecisionTreeRegressor(max_depth=5)  # or ClassifierRegressor
model.fit(X, y)
\`\`\`

Generate ONLY the Python code needed to prepare data and train the model.
The tree visualization and metrics will be handled automatically.`;

  const responseContainer = document.getElementById("llmResponse");
  document.getElementById("step3").classList.remove("d-none");

  let fullResponse = "";

  const request = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
  };

  try {
    for await (const event of asyncLLM(`${baseUrl}/chat/completions`, {
      ...request,
      body: JSON.stringify({
        model,
        stream: true,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
      }),
    })) {
      if (event.error) {
        throw new Error(event.error);
      }

      if (event.content) {
        fullResponse = event.content;
        const rendered = html`${unsafeHTML(marked.parse(fullResponse))}`;
        render(rendered, responseContainer);
      }
    }

    // Extract and execute Python code
    await extractAndExecuteCode(fullResponse);
  } catch (error) {
    throw new Error(`LLM request failed: ${error.message}`);
  }
}

// Extract and execute Python code
async function extractAndExecuteCode(response) {
  const codeMatch = response.match(/```python\n([\s\S]*?)\n```/);
  if (!codeMatch) {
    throw new Error("No Python code found in response");
  }

  currentCode = codeMatch[1].trim();

  try {
    // Convert data to Python-compatible format
    const dataRows = currentData.rows;
    const dataColumns = currentColumns;
    
    // First, set up the environment
    await pyodide.runPython(`
import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, mean_squared_error, r2_score

def tree_to_dict(tree, feature_names):
    """Convert decision tree to dictionary format."""
    def recurse(node):
        if tree.children_left[node] == _tree.TREE_LEAF:
            value = tree.value[node].flatten()
            if len(value) > 1:
                pred = float(np.argmax(value))
            else:
                pred = float(value[0])
            return {"prediction": pred}
        
        return {
            "feature": feature_names[tree.feature[node]],
            "threshold": float(tree.threshold[node]),
            "left": recurse(tree.children_left[node]),
            "right": recurse(tree.children_right[node])
        }
    
    return recurse(0)

def calculate_metrics(y_true, y_pred, is_regression=False):
    """Calculate metrics for either regression or classification."""
    if is_regression:
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        var = np.var(y_true) if np.var(y_true) != 0 else 1
        norm_mse = mse / var
        return {
            'fp': 0,
            'fn': 0,
            'precision': r2,
            'recall': 1.0 - norm_mse,
            'f1_score': r2,
            'accuracy': 1.0 - norm_mse
        }
    else:
        cm = confusion_matrix(y_true, y_pred)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)
        
        if cm.shape == (2, 2):
            fp = float(fp[1])
            fn = float(fn[1])
        else:
            fp = float(fp.sum())
            fn = float(fn.sum())
            
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)
        
        return {
            'fp': int(fp),
            'fn': int(fn),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(acc)
        }
    `);

    // Then, prepare the data
    await pyodide.runPython(`
# Load and prepare data
data_dict = json.loads('${JSON.stringify(dataRows)}')
columns_list = json.loads('${JSON.stringify(dataColumns)}')

# Create DataFrame
df = pd.DataFrame(data_dict)

# Convert all columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values with median
df = df.fillna(df.median())

print("Data preparation completed")
print("DataFrame Info:")
print(df.info())
    `);

    // Execute the user's code
    console.log("Executing user code:", currentCode);
    await pyodide.runPythonAsync(`
# Execute user code in global scope
${currentCode}

# Store X and y in global scope if not already there
if 'X' not in globals():
    X = df[df.columns[:-1]]
if 'y' not in globals():
    y = df[df.columns[-1]]

# Ensure model exists
if 'model' not in globals():
    raise ValueError("Model not found. The code must create a 'model' variable.")

# Make predictions and store in global scope
globals()['X'] = X
globals()['y'] = y
globals()['y_pred'] = model.predict(X)
globals()['feature_names'] = list(X.columns)
    `);
    
    // Finally, extract results
    const resultJson = await pyodide.runPythonAsync(`
def process_tree():
    try:
        if 'model' not in globals():
            raise ValueError("Model not found. The code must create a 'model' variable.")
        
        is_regression = isinstance(model, DecisionTreeRegressor)
        
        # Access variables from global scope
        X = globals()['X']
        y = globals()['y']
        y_pred = globals()['y_pred']
        feature_names = globals()['feature_names']
        
        # Convert tree to dictionary with explicit type conversion
        tree_dict = tree_to_dict(model.tree_, feature_names)
        metrics = calculate_metrics(y, y_pred, is_regression)
        
        # Create result dictionary with explicit type conversion
        result_dict = {
            'tree': tree_dict,
            'metrics': metrics,
            'feature_names': [str(f) for f in feature_names]  # Ensure strings
        }
        
        # Convert numpy types to Python native types
        def convert_to_native(obj):
            if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(i) for i in obj]
            return obj

        result_dict = convert_to_native(result_dict)
        
        # Verify the structure before serializing
        if not isinstance(result_dict['tree'], dict):
            raise ValueError("Tree structure is not a dictionary")
        if not isinstance(result_dict['metrics'], dict):
            raise ValueError("Metrics is not a dictionary")
        if not isinstance(result_dict['feature_names'], list):
            raise ValueError("Feature names is not a list")
        
        print("Result structure verified")
        print("Tree keys:", list(result_dict['tree'].keys()))
        print("Metrics keys:", list(result_dict['metrics'].keys()))
        
        # Serialize with error checking
        try:
            result_json = json.dumps(result_dict)
            print("JSON serialization successful")
            print("JSON length:", len(result_json))
            return result_json
        except Exception as json_error:
            print("JSON serialization failed:", str(json_error))
            print("Result dict:", result_dict)
            raise ValueError(f"Failed to serialize result: {str(json_error)}")

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("Error details:", error_details)
        raise ValueError(str(e))

# Call the function and return its result
process_tree()
`);
    
    if (!resultJson) {
        console.error("No JSON returned from Python");
        throw new Error("Failed to get result from Python - no JSON returned");
    }

    console.log("Raw result from Python:", resultJson);  // Add this line for debugging
    
    try {
        const result = JSON.parse(resultJson);
        
        if (!result) {
            throw new Error("JSON parsing resulted in null or undefined");
        }
        
        if (!result.tree || typeof result.tree !== 'object') {
            throw new Error("Missing or invalid tree structure in result");
        }
        
        if (!result.metrics || typeof result.metrics !== 'object') {
            throw new Error("Missing or invalid metrics in result");
        }
        
        if (!Array.isArray(result.feature_names)) {
            throw new Error("Missing or invalid feature names in result");
        }
        
        console.log("Result validation passed");
        console.log("Tree structure:", result.tree);
        console.log("Metrics:", result.metrics);
        console.log("Feature names:", result.feature_names);
        
        currentDecisionTree = result;
        displayDecisionTree(result.tree);
        displayMetrics(result.metrics);
        document.getElementById("step4").classList.remove("d-none");
        document.getElementById("step5").classList.remove("d-none");
        
    } catch (parseError) {
        console.error("JSON parse error:", parseError);
        console.error("Raw JSON:", resultJson);
        throw new Error(`Failed to parse Python result: ${parseError.message}`);
    }
    
  } catch (error) {
    console.error("Python execution error:", error);
    throw new Error(`Python execution failed: ${error.message}`);
  }
}

// Display decision tree
function displayDecisionTree(tree) {
  const container = document.getElementById("decisionTree");
  container.innerHTML = renderTreeCollapsible(tree);
}

// Display metrics
function displayMetrics(metrics) {
  const container = document.getElementById("metricsGrid");
  const metricsData = [
    { label: "Accuracy", value: (metrics.accuracy || 0).toFixed(3) },
    { label: "Precision", value: (metrics.precision || 0).toFixed(3) },
    { label: "Recall", value: (metrics.recall || 0).toFixed(3) },
    { label: "F1 Score", value: (metrics.f1_score || 0).toFixed(3) },
    { label: "False Positives", value: metrics.fp || 0 },
    { label: "False Negatives", value: metrics.fn || 0 },
  ];

  const metricsHtml = metricsData
    .map(
      (metric) => `
    <div class="col-lg-2 col-md-4 col-6">
      <div class="bg-light p-3 rounded text-center">
        <div class="fs-4 fw-bold text-primary">${metric.value}</div>
        <div class="small text-muted">${metric.label}</div>
      </div>
    </div>
  `,
    )
    .join("");

  container.innerHTML = metricsHtml;
}

// Revise analysis
async function reviseAnalysis() {
  const revisionPrompt = document.getElementById("revisionPrompt").value;
  if (!revisionPrompt.trim()) {
    showError("Please enter revision instructions.");
    return;
  }

  const originalPrompt = document.getElementById("promptText").value;
  const combinedPrompt = `${originalPrompt}\n\nAdditional instructions: ${revisionPrompt}`;

  document.getElementById("promptText").value = combinedPrompt;
  await analyzeData();
}

// Utility functions
function showLoading(show) {
  document.getElementById("loadingOverlay").classList.toggle("d-none", !show);
}

function showError(message) {
  const alertHtml = `
    <div class="alert alert-danger alert-dismissible fade show" role="alert">
      <i class="bi bi-exclamation-triangle me-2"></i>
      ${message}
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    </div>
  `;
  document.body.insertAdjacentHTML("afterbegin", alertHtml);
}

function markdownToHtml(markdown) {
  if (!markdown) return "";
  return marked.parse(markdown);
}

function renderTreeCollapsible(node) {
  if (node.prediction !== undefined) {
    return `<div class="alert alert-success border-success text-break">
      <i class="bi bi-check-circle-fill me-2"></i>
      <strong>Prediction:</strong> <span class="badge bg-success ms-2">${node.prediction}</span>
    </div>`;
  }
  
  const threshold = typeof node.threshold === "number" ? node.threshold.toFixed(3) : node.threshold;
  
  return `
    <details class="mb-2">
      <summary class="text-break fw-bold text-primary cursor-pointer" style="cursor: pointer;">
        <i class="bi bi-diagram-3 me-2"></i>
        Is <strong>${node.feature}</strong> &lt; <strong>${threshold}</strong>?
      </summary>
      <div class="ms-4 mt-2">
        <div class="mb-2">
          <span class="badge bg-success me-2">YES</span>
          ${renderTreeCollapsible(node.left)}
        </div>
        <div>
          <span class="badge bg-danger me-2">NO</span>
          ${renderTreeCollapsible(node.right)}
        </div>
      </div>
    </details>
  `;
}
