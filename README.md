# Decision Tree Builder

An interactive web application for building and visualizing decision trees from your data. Built with JavaScript and Python (via Pyodide), this tool helps you analyze datasets and create decision trees with an intuitive interface.

## Features

- **Interactive Data Upload**: Support for CSV and XLSX file formats
- **Sample Datasets**: Pre-configured datasets for quick experimentation
- **Modern UI**: Clean, responsive Bootstrap 5 interface with intuitive step-by-step workflow
- **Interactive Visualization**: Easy-to-read decision tree output
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

3. **Configure Analysis**:
   - Select your target column
   - Review and modify the analysis prompt if needed

4. **Generate Tree**:
   - Click "Analyze" to build your decision tree
   - View the generated Python code in the collapsible panel
   - Examine the decision tree visualization

5. **Review Results**:
   - Examine model performance metrics
   - Understand the decision paths

6. **Refine Analysis** (Optional):
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
- **AI Integration**:
  - LLM Foundry API

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