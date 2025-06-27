import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.mjs";
import { showError, showSuccess, showInfo, showLoading } from './utils.js';

let pyodide = null;

export async function initializePyodide() {
  try {
    showLoading(true);
    pyodide = await loadPyodide();
    await pyodide.loadPackage(["pandas", "scikit-learn", "numpy"]);
    showInfo("Python environment initialized successfully!");
    await pyodide.runPython(`
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
    `);
  } catch (error) {
    showError("Failed to initialize Python environment: " + error.message);
  } finally {
    showLoading(false);
  }
}

export async function extractAndExecuteCode(response, currentData, currentColumns, derivedMetricsOnly = false) {
  const codeMatch = response.match(/```python\n([\s\S]*?)\n```/);
  if (!codeMatch) throw new Error("No Python code found in response");

  const currentCode = codeMatch[1].trim();

  try {
    const dataRows = currentData.rows;
    const dataColumns = currentColumns;
    
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

    await pyodide.runPython(`
# Load and prepare data
data_dict = json.loads('${JSON.stringify(dataRows)}')
columns_list = json.loads('${JSON.stringify(dataColumns)}')

# Create DataFrame
df = pd.DataFrame(data_dict)

# Convert numeric columns only, preserve categorical
numeric_cols = df.select_dtypes(include=['object']).columns
for col in numeric_cols:
    # Try to convert to numeric, but only if it makes sense
    converted = pd.to_numeric(df[col], errors='coerce')
    if not converted.isna().all():  # Only convert if at least some values are numeric
        df[col] = converted

# Handle missing values more carefully
if df.empty:
    raise ValueError("DataFrame is empty after processing")
else:
    # Fill NaN values only if there are valid values to compute median/mode
    for col in df.columns:
        if df[col].isna().all():
            df[col] = 0 if df[col].dtype in ['float64', 'int64'] else 'Unknown'
        elif df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')

# Encode categorical variables for machine learning
from sklearn.preprocessing import LabelEncoder
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Only drop rows if we still have data
if not df.empty:
    initial_rows = len(df)
    df = df.dropna()

# Execute user code
exec("""
try:
${currentCode.split('\n').map(line => '    ' + line).join('\n')}
except TypeError as e:
    if "Cannot convert" in str(e) and "to numeric" in str(e):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Re-run the corrected code
${currentCode.replace('df.fillna(df.median(), inplace=True)', 'numeric_cols = df.select_dtypes(include=[np.number]).columns; df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())').split('\n').map(line => '        ' + line).join('\n')}
    else:
        raise
""")

${derivedMetricsOnly ? `
# For derived metrics only, return the enhanced dataframe
enhanced_data = {
    'headers': list(df.columns),
    'rows': df.to_dict('records')
}
` : `
# Store X and y in global scope if not already there
if 'X' not in globals():
    X = df[df.columns[:-1]]
if 'y' not in globals():
    y = df[df.columns[-1]]

# Validate data before model training
if len(X) == 0:
    raise ValueError("No samples available for training. Check your data preprocessing.")
if y.isna().any():
    raise ValueError("Target variable contains NaN values. Check your data cleaning.")

# Ensure model exists
if 'model' not in globals():
    raise ValueError("Model not found. The code must create a 'model' variable.")

# Make predictions and store in global scope
globals()['X'] = X
globals()['y'] = y
globals()['y_pred'] = model.predict(X)
globals()['feature_names'] = list(X.columns)
`}
    `);
    
    let resultJson;
    if (derivedMetricsOnly) {
      resultJson = await pyodide.runPythonAsync(`
import json
json.dumps(enhanced_data)
`);
    } else {
      resultJson = await pyodide.runPythonAsync(`
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
        
        # Serialize with error checking
        try:
            result_json = json.dumps(result_dict)
            return result_json
        except Exception as json_error:
            raise ValueError(f"Failed to serialize result: {str(json_error)}")

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise ValueError(str(e))

# Call the function and return its result
process_tree()
`);
    }
    
    if (!resultJson) {
        throw new Error("Failed to get result from Python - no JSON returned");
    }
    
    try {
        const result = JSON.parse(resultJson);
        
        if (!result) {
            throw new Error("JSON parsing resulted in null or undefined");
        }
        
        if (derivedMetricsOnly) {
            // For derived metrics, return the enhanced data structure
            return { enhancedData: result };
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
        
        return result;
        
    } catch (parseError) {
        throw new Error(`Failed to parse Python result: ${parseError.message}`);
    }
    
  } catch (error) {
    throw new Error(`Python execution failed: ${error.message}`);
  }
}