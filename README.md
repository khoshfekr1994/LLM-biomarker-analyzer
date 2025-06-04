# Interactive Data Analyzer

🔬 An AI-powered interactive data analysis tool that uses Ollama to generate and execute Python analysis code through natural language conversations.

## Features

- 🤖 **Conversational Analysis**: Ask questions in natural language and get executable Python code
- 📊 **Biomarker Analysis**: Built-in helper functions for biomarker research and ROC analysis
- 📈 **Comprehensive Visualization**: Automatic generation of plots and charts
- 🧬 **Statistical Analysis**: T-tests, correlations, feature selection, and machine learning
- 📁 **Multiple Formats**: Supports CSV and Excel files
- 💬 **Interactive Chat**: Persistent conversation with context awareness

## Prerequisites

### 1. Install Ollama
```bash
# Install Ollama (visit https://ollama.ai for OS-specific instructions)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (recommended: llama3.2)
ollama pull llama3.2
```

### 2. Python Requirements
- Python 3.8+
- Required packages (see requirements.txt)

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/khoshfekr1994/LLM-biomarker-analyzer.git
cd LLM-biomarker-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start Ollama service**
```bash
ollama serve
```

## Quick Start

### Option 1: Direct Run
```python
python src/LLM_biomarker_analyzer.py
```

### Option 2: Import in Your Code
```python
from src.LLM_biomarker_analyzer import InteractiveDataAnalyzer

# With file path
analyzer = InteractiveDataAnalyzer("your_data.csv")
analyzer.start_chat()

# Or let it prompt for file
analyzer = InteractiveDataAnalyzer()
analyzer.start_chat()
```

## Usage Examples

### Basic Analysis Questions
```
🧑‍🔬 Your Question: What are the top 5 most important biomarkers?
🧑‍🔬 Your Question: Create ROC curves for cancer detection
🧑‍🔬 Your Question: Show correlation heatmap of all variables
🧑‍🔬 Your Question: Build a machine learning model to predict outcomes
```

### Biomarker-Specific Analysis
```
🧑‍🔬 Your Question: Calculate AUC scores for each protein marker
🧑‍🔬 Your Question: Compare control vs cancer groups using t-tests
🧑‍🔬 Your Question: Find outliers in the protein expression data
```

### Available Commands
- `help` - Show command examples
- `info` - Display dataset information
- `upload` - Load a new dataset
- `history` - View conversation history
- `clear` - Clear conversation history
- `quit` - Exit the analyzer

## Built-in Helper Functions

### `clean_numeric_column(df, col_name)`
Converts columns to numeric format, handling common data issues.

### `calculate_auc_binary(df, group_col, group1, group2, biomarker_col)`
Calculates AUC for binary classification between two groups.

## Dataset Format

The analyzer works best with datasets containing:
- **Group/Label columns**: e.g., 'Tumor_type', 'Diagnosis'
- **Biomarker columns**: Numeric values (can be stored as text)
- **Sample identifiers**: Patient IDs, sample names

Example CSV structure:
```csv
Sample_ID,Tumor_type,Biomarker1,Biomarker2,Biomarker3
S001,Control,1.23,0.45,2.67
S002,Cancer,2.34,0.78,3.45
...
```

## Configuration

### Change Ollama Model
```python
analyzer = InteractiveDataAnalyzer(model="llama3.1")  # or other models
```

### Custom Analysis Functions
Add your own helper functions to the `globals_dict` in the `__init__` method.

## Troubleshooting

### Common Issues

1. **"Error connecting to Ollama"**
   - Ensure Ollama is running: `ollama serve`
   - Check if the model is installed: `ollama list`

2. **"Column not found" errors**
   - Check column names: `df.columns`
   - Use exact column names including spaces/special characters

3. **Data type issues**
   - Use `clean_numeric_column()` for numeric conversions
   - Check data types: `df.dtypes`

### Getting Help
- Type `help` in the chat for examples
- Type `info` to see dataset information
- Check the [installation guide](docs/installation.md)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Ollama](https://ollama.ai) for local LLM inference
- Uses scientific Python ecosystem (pandas, scikit-learn, scipy)
- Inspired by the need for conversational data analysis in research

## Example Output

```
🔬 Interactive Data Analysis Chat
==================================================
📊 Dataset loaded successfully!
📊 Shape: 150 rows, 25 columns

🧑‍🔬 Your Question: Compare cancer vs control biomarkers

📝 Generated Analysis Code:
==========================================
🚀 Executing Analysis Block 1:
# Statistical comparison of cancer vs control groups
cancer_data = df[df['Tumor_type'] == 'Cancer']
control_data = df[df['Tumor_type'] == 'Control']

print(f"Cancer samples: {len(cancer_data)}")
print(f"Control samples: {len(control_data)}")
✅ Block 1 completed successfully
==========================================
```
