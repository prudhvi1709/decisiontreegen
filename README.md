# Decision Tree Builder

An interactive web application for building and visualizing decision trees from your data. Built with JavaScript and Python (via Pyodide), this tool helps you analyze datasets and create decision trees with an intuitive interface.

## Features

- **Interactive Data Upload**: Support for CSV and XLSX file formats
- **Sample Datasets**: Pre-configured datasets for quick experimentation
- **AI-Powered Derived Metrics**: Generate meaningful derived features from your data using AI
- **Granular Feature Selection**: Choose exactly which derived metrics to include in your decision tree
- **Interactive Tree Analysis**: Ask natural language questions about your decision tree and get visual highlights
- **Smart Tree Highlighting**: Automatically highlights relevant decision paths and nodes based on your questions
- **Modern UI**: Clean, responsive Bootstrap 5 interface with intuitive step-by-step workflow
- **Interactive Visualization**: Easy-to-read decision tree output with expandable/collapsible nodes
- **Performance Metrics**: View model evaluation metrics including accuracy, precision, and more
- **Revision System**: Ability to refine and improve your analysis
- **Python Integration**: Seamless Python execution in the browser using Pyodide
- **LLM Integration**: Powered by LLM Foundry for intelligent data analysis

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/prudhvi1709/decisiontreegen.git
   cd decisiontreegen
   ```

2. Open `index.html` in a modern web browser.

3. Configure your API settings in the advanced settings panel.

## Using the Application

1. **Configure Settings** (Optional):
   - Set your API key in the advanced settings
   - Choose your preferred base URL
   - Select the LLM model to use

2. **Upload Data**:
   - Use the file upload button to select your CSV/XLSX file
   - Or choose from the available sample datasets
   - Preview your data before proceeding

3. **Generate Derived Metrics** (Optional):
   - Enable AI-powered derived metrics generation
   - Select which columns to use for creating derived features
   - Add custom metric definitions if desired
   - Choose whether to include derived metrics in the decision tree
   - Click "Generate Derived Metrics" to create AI-powered features
   - Preview the enhanced dataset with new derived columns
   - Select exactly which derived metrics to include in your analysis

4. **Configure Analysis**:
   - Select your target column from original or enhanced dataset
   - Review and modify the analysis prompt if needed

5. **Generate Tree**:
   - Click "Analyze" to build your decision tree
   - View the generated Python code in the collapsible panel
   - Examine the decision tree visualization

6. **Review Results**:
   - Examine model performance metrics
   - Understand the decision paths with original and derived features

7. **Interactive Tree Analysis**:
   - Ask questions about your decision tree in natural language
   - Examples: "What is the major impact on high prices?", "Which features lead to positive outcomes?"
   - View AI-powered explanations and visual highlights of relevant tree sections
   - Automatically highlights decision nodes and prediction outcomes
   - Clear highlights and ask new questions for different insights

8. **Refine Analysis** (Optional):
   - Add revision instructions
   - Click "Revise Analysis" to improve the model

## Advanced Settings

- **API Configuration**: 
  - OpenAI API Key setup
  - Configurable base URL with preset options
- **Model Selection**: Choose from available models:
  - gpt-4.1-nano
  - gpt-4.1-mini
  - gpt-4o-mini
  - o3-mini
- **Tree Parameters**: Control tree depth (1-20)

## AI-Powered Derived Metrics

This application features intelligent derived metrics generation that enhances your dataset with meaningful engineered features:

### What are Derived Metrics?
Derived metrics are new features automatically created from your existing data columns using AI reasoning. These can include:
- **Mathematical operations**: Ratios, products, differences between columns
- **Statistical transformations**: Logarithmic, square root, standardization
- **Domain-specific calculations**: BMI from height/weight, efficiency ratios, etc.
- **Interaction features**: Combinations of related columns

### How It Works
1. **Column Selection**: Choose which original columns to use for generating derived features
2. **Custom Metrics**: Optionally define your own derived metrics using natural language
3. **AI Generation**: The LLM analyzes your data context and creates meaningful derived features
4. **Preview & Selection**: Review all generated metrics and choose which ones to include in your decision tree
5. **Intelligent Integration**: Selected metrics are seamlessly incorporated into the model training

### Benefits
- **Enhanced Model Performance**: More features can lead to better decision tree accuracy
- **Automatic Feature Engineering**: No manual calculation of complex derived features
- **Data Insights**: Discover relationships and patterns you might have missed
- **Full Control**: Choose exactly which derived metrics to use in your analysis

## Interactive Tree Analysis

Transform your decision tree analysis with AI-powered natural language queries and visual highlighting:

### What You Can Ask
Ask questions in plain English about your decision tree to gain deeper insights:
- **Feature Importance**: "What is the major impact on high prices?"
- **Decision Paths**: "Which features lead to positive outcomes?"
- **Model Behavior**: "What causes low accuracy predictions?"
- **Specific Features**: "How important is [feature_name] in the model?"
- **Outcome Analysis**: "What leads to the best results?"

### Visual Highlighting System
- **Automatic Highlighting**: Relevant decision nodes are highlighted in yellow with pulsing animations
- **Prediction Highlighting**: Important prediction outcomes are highlighted in blue
- **Auto-Expansion**: Relevant tree sections automatically expand for easy viewing
- **Smart Scrolling**: Automatically scrolls to the most relevant parts of the tree
- **Clear & Re-analyze**: Ask multiple questions with automatic highlight clearing

### How It Works
1. **Natural Language Processing**: AI analyzes your question in the context of your decision tree
2. **Intelligent Path Finding**: Identifies the most relevant decision paths and features
3. **Visual Enhancement**: Highlights important nodes with smooth animations and clear borders
4. **Contextual Explanations**: Provides detailed explanations and key insights about the highlighted areas
5. **Interactive Exploration**: Ask follow-up questions to explore different aspects of your model

### Example Queries
```
"What is the major impact on high sales?"
→ Highlights all decision nodes that significantly affect sales outcomes

"Which path leads to the best customer satisfaction?"
→ Shows the decision sequence that results in highest satisfaction scores

"How does price affect the final decision?"
→ Highlights all nodes where price is used as a decision criterion
```

## Performance Metrics

The application provides key model evaluation metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- False Positives
- False Negatives

## Technical Stack

- **Frontend**: 
  - HTML5
  - Bootstrap 5
  - JavaScript (ES6+)
  - Bootstrap Icons
- **Machine Learning**: 
  - Python (via Pyodide)
  - scikit-learn
- **Data Processing**: 
  - Pandas
  - NumPy
  - Automated categorical encoding
- **AI Integration**:
  - LLM Foundry API
  - AI-powered derived metrics generation
  - Intelligent feature engineering
  - Natural language tree analysis
- **Visualization**:
  - Lit-HTML for performant rendering
  - CSS animations and transitions
  - Interactive tree highlighting

## Security

- API keys are stored locally in browser storage
- No data is sent to external servers except the LLM API
- All processing happens in your browser using Pyodide

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please open an issue in the GitHub repository. 