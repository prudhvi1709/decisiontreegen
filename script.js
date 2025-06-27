import { asyncLLM } from "https://cdn.jsdelivr.net/npm/asyncllm@2";
import {showLoading, showError, showSuccess, showInfo, showWarning, setupFormPersistence, displayMetrics, displayDecisionTree, renderMarkdownResponse, setRelevantPaths} from "./utils.js";
import { handleFileUpload, loadSampleDataset, displayDataPreview, populateTargetColumn } from "./data.js";
import { initializePyodide, extractAndExecuteCode } from "./pyworker.js";

let currentData = null,
  currentColumns = [],
  enhancedData = null,
  derivedMetricsColumns = [],
  currentDecisionTree = null,
  config = null;

async function loadConfig() {
  const response = await fetch("config.json");
  config = await response.json();
  const { baseUrl, model } = config.defaultSettings;
  document.getElementById("baseUrlInput").value = baseUrl;
  document.getElementById("modelInput").value = model;
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
  document.getElementById("analyzeBtn").addEventListener("click", (e) => {
    e.preventDefault();
    analyzeData().catch((error) =>
      showError("Analysis failed: " + error.message)
    );
  });
  document
    .getElementById("reviseBtn")
    .addEventListener("click", reviseAnalysis);
  document
    .getElementById("enableDerivedMetrics")
    .addEventListener("change", toggleDerivedMetrics);
  document
    .getElementById("generateDerivedBtn")
    .addEventListener("click", generateDerivedMetrics);
  document
    .getElementById("skipDerivedBtn")
    .addEventListener("click", skipToDTAnalysis);
  document
    .getElementById("proceedToAnalysisBtn")
    .addEventListener("click", proceedToAnalysis);
  document
    .getElementById("selectAllDerived")
    .addEventListener("click", selectAllDerivedMetrics);
  document
    .getElementById("selectNoneDerived")
    .addEventListener("click", selectNoneDerivedMetrics);
  document
    .getElementById("analyzeTreeBtn")
    .addEventListener("click", analyzeTreeQuery);
  document
    .getElementById("clearHighlightBtn")
    .addEventListener("click", clearTreeHighlights);
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
  enhancedData = null;
  displayDataPreview(data);
  populateColumnCheckboxes(data.headers);
  document.getElementById("step2").classList.remove("d-none");
  document.getElementById("step2").scrollIntoView({ behavior: "smooth" });
}

function toggleDerivedMetrics() {
  const isEnabled = document.getElementById("enableDerivedMetrics").checked;
  document.getElementById("derivedMetricsSection").classList.toggle("d-none", !isEnabled);
}

function populateColumnCheckboxes(headers) {
  const container = document.getElementById("columnCheckboxes");
  container.innerHTML = headers
    .map(
      (col) => `
    <div class="form-check">
      <input class="form-check-input" type="checkbox" value="${col}" id="col_${col}" checked>
      <label class="form-check-label" for="col_${col}">${col}</label>
    </div>
  `
    )
    .join("");
}

async function generateDerivedMetrics() {
  const apiKey = document.getElementById("apiKeyInput").value;
  const baseUrl = document.getElementById("baseUrlInput").value;
  const model = document.getElementById("modelInput").value;
  const selectedColumns = Array.from(document.querySelectorAll('#columnCheckboxes input:checked')).map(cb => cb.value);
  const customMetrics = document.getElementById("customMetrics").value.trim();

  if (!apiKey) return showError("Please enter your OpenAI API key in the advanced settings.");
  if (!currentData) return showError("Please upload a dataset first.");
  if (selectedColumns.length === 0) return showError("Please select at least one column for derived metrics.");

  try {
    showLoading(true);
    const derivedMetricsData = await generateDerivedMetricsWithAI(baseUrl, apiKey, model, selectedColumns, customMetrics);
    enhancedData = derivedMetricsData;
    displayEnhancedDataPreview(derivedMetricsData);
    showSuccess(`Successfully generated ${derivedMetricsColumns.length} derived metrics!`);
    document.getElementById("step2_5").classList.remove("d-none");
    document.getElementById("step2_5").scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    showError("Failed to generate derived metrics: " + error.message);
  } finally {
    showLoading(false);
  }
}

function skipToDTAnalysis() {
  enhancedData = currentData;
  showInfo("Skipped derived metrics generation. Proceeding with original dataset.");
  proceedToAnalysis();
}

function proceedToAnalysis() {
  const dataToUse = enhancedData || currentData;
  populateTargetColumn(dataToUse.headers);
  updatePrompt();
  document.getElementById("step3").classList.remove("d-none");
  document.getElementById("step3").scrollIntoView({ behavior: "smooth" });
}

function displayEnhancedDataPreview(data) {
  const table = document.getElementById("enhancedPreviewTable");
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  
  thead.innerHTML = `<tr>${data.headers.map(h => `<th>${h}</th>`).join("")}</tr>`;
  tbody.innerHTML = data.rows.slice(0, 10).map(row => 
    `<tr>${data.headers.map(h => `<td>${row[h] || ''}</td>`).join("")}</tr>`
  ).join("");
  
  // Identify derived metrics (new columns not in original data)
  derivedMetricsColumns = data.headers.filter(h => !currentColumns.includes(h));
  populateDerivedMetricsSelection();
}

function populateDerivedMetricsSelection() {
  const container = document.getElementById("derivedMetricsSelection");
  if (derivedMetricsColumns.length === 0) {
    container.innerHTML = '<div class="col-12 text-muted">No derived metrics generated</div>';
    return;
  }
  
  container.innerHTML = derivedMetricsColumns
    .map(col => `
      <div class="col-md-6 col-lg-4 mb-2">
        <div class="form-check">
          <input class="form-check-input" type="checkbox" value="${col}" id="derived_${col}" checked>
          <label class="form-check-label" for="derived_${col}">
            <small><strong>${col}</strong></small>
          </label>
        </div>
      </div>
    `).join("");
}

function selectAllDerivedMetrics() {
  document.querySelectorAll('#derivedMetricsSelection input[type="checkbox"]').forEach(cb => cb.checked = true);
  showInfo("All derived metrics selected for decision tree.");
}

function selectNoneDerivedMetrics() {
  document.querySelectorAll('#derivedMetricsSelection input[type="checkbox"]').forEach(cb => cb.checked = false);
  showInfo("All derived metrics deselected.");
}

function getSelectedDerivedMetrics() {
  return Array.from(document.querySelectorAll('#derivedMetricsSelection input:checked')).map(cb => cb.value);
}

async function analyzeTreeQuery() {
  const query = document.getElementById("treeQuery").value.trim();
  if (!query) {
    showError("Please enter a question about the decision tree.");
    return;
  }
  
  if (!currentDecisionTree) {
    showError("No decision tree available to analyze.");
    return;
  }
  
  const apiKey = document.getElementById("apiKeyInput").value;
  const baseUrl = document.getElementById("baseUrlInput").value;
  const model = document.getElementById("modelInput").value;
  
  if (!apiKey) {
    showError("Please enter your OpenAI API key in the advanced settings.");
    return;
  }
  
  try {
    showLoading(true);
    // Clear previous highlights and get new analysis
    setRelevantPaths([]);
    const analysis = await getTreeAnalysis(baseUrl, apiKey, model, query, currentDecisionTree);
    displayTreeAnalysis(analysis);
    // Set relevant paths and re-render tree with highlights
    setRelevantPaths(analysis.relevantPaths);
    displayDecisionTree(currentDecisionTree.tree);
    showInfo("Tree analysis complete! Relevant nodes are highlighted below.");
    // Scroll to first highlighted element
    setTimeout(() => {
      const firstHighlighted = document.querySelector('.tree-highlighted');
      if (firstHighlighted) {
        firstHighlighted.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 100);
  } catch (error) {
    showError("Failed to analyze tree: " + error.message);
  } finally {
    showLoading(false);
  }
}

async function getTreeAnalysis(baseUrl, apiKey, model, query, treeData) {
  const systemPrompt = `You are a decision tree analysis expert. Analyze the given decision tree structure and answer the user's question by identifying relevant paths and nodes.

Decision Tree Structure: ${JSON.stringify(treeData.tree)}
Feature Names: ${treeData.feature_names.join(", ")}
Model Metrics: ${JSON.stringify(treeData.metrics)}

Your task is to:
1. Understand the user's question about the decision tree
2. Identify which nodes, features, and decision paths are most relevant
3. Provide a clear explanation
4. Return specific node paths that should be highlighted

Please return a JSON response with this structure:
{
  "explanation": "Clear explanation answering the user's question",
  "relevantPaths": ["path descriptions or feature names that are most relevant"],
  "keyInsights": ["important insights about the tree structure"]
}

Focus on identifying:
- Important features that drive decisions
- Critical decision points
- Paths leading to specific outcomes
- Feature importance relative to the question`;

  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: query },
      ],
    }),
  });

  const data = await response.json();
  const content = data.choices[0].message.content;
  
  // Try to extract JSON from the response
  try {
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }
  } catch (e) {
    // Fallback if JSON parsing fails
  }
  
  return {
    explanation: content,
    relevantPaths: [],
    keyInsights: []
  };
}

function displayTreeAnalysis(analysis) {
  const container = document.getElementById("treeAnalysisResult");
  const textContainer = document.getElementById("treeAnalysisText");
  
  let content = `<p><strong>Answer:</strong> ${analysis.explanation}</p>`;
  
  if (analysis.keyInsights && analysis.keyInsights.length > 0) {
    content += `<p><strong>Key Insights:</strong></p><ul>`;
    analysis.keyInsights.forEach(insight => {
      content += `<li>${insight}</li>`;
    });
    content += `</ul>`;
  }
  
  textContainer.innerHTML = content;
  container.classList.remove("d-none");
}

function clearTreeHighlights() {
  // Clear relevant paths and re-render tree without highlights
  setRelevantPaths([]);
  if (currentDecisionTree) {
    displayDecisionTree(currentDecisionTree.tree);
  }
  showInfo("Tree highlights cleared.");
  
  // Hide analysis result
  document.getElementById("treeAnalysisResult").classList.add("d-none");
}

function updatePrompt() {
  const target = document.getElementById("targetColumn").value;
  document.getElementById(
    "promptText"
  ).value = `Predict ${target} using a decision tree with max_depth=5.`;
}

async function generateDerivedMetricsWithAI(baseUrl, apiKey, model, selectedColumns, customMetrics) {
  const systemPrompt = `You are a data science expert. Generate meaningful derived metrics from the given dataset columns.

Dataset columns: ${currentColumns.join(", ")}
Selected columns for derived metrics: ${selectedColumns.join(", ")}
${customMetrics ? `Custom metrics requested: ${customMetrics}` : ""}
Data sample: ${JSON.stringify(currentData.rows.slice(0, 3))}

Generate Python code that:
1. Creates meaningful derived features from the selected columns
2. Includes mathematical operations (ratios, products, differences)
3. Includes statistical transformations where appropriate
4. Returns the enhanced DataFrame with new columns

The DataFrame 'df' is already created. Return ONLY Python code that adds derived columns to df.

Example:
\`\`\`python
# Generate derived metrics
df['bmi'] = df['weight'] / (df['height'] ** 2)
df['income_ratio'] = df['income'] / df['expenses']
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 50, 75, 100], labels=['Young', 'Adult', 'Middle', 'Senior'])
\`\`\``;

  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: "Generate derived metrics code" },
      ],
    }),
  });

  const data = await response.json();
  const code = data.choices[0].message.content;
  
  // Execute the derived metrics code and return enhanced data
  const result = await extractAndExecuteCode(code, currentData, currentColumns, true);
  return result.enhancedData || currentData;
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
  if (!enhancedData && !currentData) return showError("Please upload a dataset first.");

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
  const dataToUse = enhancedData || currentData;
  const selectedDerivedMetrics = enhancedData ? getSelectedDerivedMetrics() : [];
  const excludedDerivedMetrics = enhancedData ? derivedMetricsColumns.filter(col => !selectedDerivedMetrics.includes(col)) : [];
  
  let columnsInstructions = "";
  if (enhancedData && excludedDerivedMetrics.length > 0) {
    columnsInstructions = `\n- Exclude these derived metric columns from the model training: ${excludedDerivedMetrics.join(", ")}`;
  }
  
  const systemPrompt = `You are a data science expert. Generate Python code to create a decision tree model for the given dataset.

Dataset columns: ${dataToUse.headers.join(", ")}
Data sample: ${JSON.stringify(dataToUse.rows.slice(0, 3))}

Your task is to:
1. Prepare the data for training
2. Create and train a decision tree model

The DataFrame 'df' is already created and available${enhancedData ? " with derived metrics" : ""}. Your code should:
1. Handle any missing values in the data
2. Select appropriate features and target columns
3. Create and train a decision tree model named 'model'

Notes:
- All necessary imports are already done (pandas, numpy, sklearn)
- Use DecisionTreeRegressor for numeric targets
- Use DecisionTreeClassifier for categorical targets
- Set max_depth=5 as the default hyperparameter${columnsInstructions}

Example:
\`\`\`python
# Handle missing values
df.fillna(df.median(), inplace=True)

# Select target and features
target_col = 'price'  # choose appropriate target
feature_cols = [col for col in df.columns if col != target_col]
${excludedDerivedMetrics.length > 0 ? `
# Exclude specific derived metrics
excluded_cols = ${JSON.stringify(excludedDerivedMetrics)}
feature_cols = [col for col in feature_cols if col not in excluded_cols]
` : ""}
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
  document.getElementById("step4").classList.remove("d-none");

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
      dataToUse,
      dataToUse.headers
    );
    currentDecisionTree = result;
    displayDecisionTree(result.tree);
    displayMetrics(result.metrics);
    showSuccess(`Decision tree model created successfully! Accuracy: ${(result.metrics.accuracy * 100).toFixed(1)}%`);
    document.getElementById("step5").classList.remove("d-none");
    document.getElementById("step6").classList.remove("d-none");
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
