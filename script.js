import { asyncLLM } from "https://cdn.jsdelivr.net/npm/asyncllm@2";
import {showLoading, showError, setupFormPersistence, displayMetrics, displayDecisionTree, renderMarkdownResponse} from "./utils.js";
import { handleFileUpload, loadSampleDataset, displayDataPreview, populateTargetColumn } from "./data.js";
import { initializePyodide, extractAndExecuteCode } from "./pyworker.js";

let currentData = null,
  currentColumns = [],
  currentDecisionTree = null,
  config = null;

async function loadConfig() {
  const response = await fetch("config.json");
  config = await response.json();
  const { baseUrl, model, maxDepth } = config.defaultSettings;
  document.getElementById("baseUrlInput").value = baseUrl;
  document.getElementById("modelInput").value = model;
  document.getElementById("maxDepth").value = maxDepth;
}

document.addEventListener("DOMContentLoaded", async () => {
  await loadConfig();
  initializePyodide();
  setupEventListeners();
  renderSampleDatasets();
  setupFormPersistence();
});

function setupEventListeners() {
  document
    .getElementById("fileInput")
    .addEventListener("change", (e) => handleFileUpload(e, processData));
  document
    .getElementById("targetColumn")
    .addEventListener("change", updatePrompt);
  document.getElementById("maxDepth").addEventListener("change", updatePrompt);
  document.getElementById("analyzeBtn").addEventListener("click", (e) => {
    e.preventDefault();
    analyzeData().catch((error) =>
      showError("Analysis failed: " + error.message)
    );
  });
  document
    .getElementById("reviseBtn")
    .addEventListener("click", reviseAnalysis);
}

function renderSampleDatasets() {
  const container = document.getElementById("sampleDatasets");
  if (!config?.sampleDatasets) return;

  container.innerHTML = config.sampleDatasets
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
    </div>`
    )
    .join("");
}

window.loadSampleDataset = (url, target) =>
  loadSampleDataset(url, target, processData);

function processData(data, defaultTarget = null) {
  currentData = data;
  currentColumns = data.headers;
  displayDataPreview(data);
  populateTargetColumn(data.headers, defaultTarget);
  updatePrompt();
  document.getElementById("step2").classList.remove("d-none");
  document.getElementById("step2").scrollIntoView({ behavior: "smooth" });
}

function updatePrompt() {
  const target = document.getElementById("targetColumn").value;
  const depth = document.getElementById("maxDepth").value;
  document.getElementById(
    "promptText"
  ).value = `Predict ${target} using a decision tree. Max depth: ${depth}.`;
}

async function analyzeData() {
  const apiKey = document.getElementById("apiKeyInput").value;
  const baseUrl = document.getElementById("baseUrlInput").value;
  const model = document.getElementById("modelInput").value;
  const prompt = document.getElementById("promptText").value;

  if (!apiKey)
    return showError(
      "Please enter your OpenAI API key in the advanced settings."
    );
  if (!currentData) return showError("Please upload a dataset first.");

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
Data sample: ${
    currentData && currentData.rows
      ? JSON.stringify(currentData.rows.slice(0, 3))
      : "No data"
  }

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
        renderMarkdownResponse(fullResponse, responseContainer);
      }
    }

    // Extract and execute Python code
    const result = await extractAndExecuteCode(
      fullResponse,
      currentData,
      currentColumns
    );
    currentDecisionTree = result;
    displayDecisionTree(result.tree);
    displayMetrics(result.metrics);
    document.getElementById("step4").classList.remove("d-none");
    document.getElementById("step5").classList.remove("d-none");
  } catch (error) {
    throw new Error(`LLM request failed: ${error.message}`);
  }
}
async function reviseAnalysis() {
  const revisionPrompt = document.getElementById("revisionPrompt").value;
  if (!revisionPrompt.trim())
    return showError("Please enter revision instructions.");

  const originalPrompt = document.getElementById("promptText").value;
  document.getElementById(
    "promptText"
  ).value = `${originalPrompt}\n\nAdditional instructions: ${revisionPrompt}`;
  await analyzeData();
}
