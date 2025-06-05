import pandas as pd
import numpy as np
import requests
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

class InteractiveDataAnalyzer:
    def __init__(self, file_path=None, model="llama3.2"):
        self.file_path = file_path
        self.model = model
        self.df = None
        self.conversation_history = []
        # Create a persistent globals dict with all imports
        self.globals_dict = {
            'pd': pd, 'pandas': pd, 'np': np, 'numpy': np, 
            'plt': plt, 'matplotlib': plt, 'sns': sns, 'seaborn': sns,
            'stats': stats, 'SelectKBest': SelectKBest, 'f_classif': f_classif,
            'StandardScaler': StandardScaler, 'train_test_split': train_test_split,
            'RandomForestClassifier': RandomForestClassifier, 'roc_auc_score': roc_auc_score,
            'roc_curve': roc_curve, '__builtins__': __builtins__,
            # Add more imports that might be useful
            'warnings': warnings
        }
        
        # Add helper functions for common biomarker analysis
        def clean_numeric_column(df, col_name):
            """Convert a column to numeric, handling common issues"""
            if col_name not in df.columns:
                print(f"Column '{col_name}' not found")
                return None
            
            # Try to convert to numeric
            numeric_col = pd.to_numeric(df[col_name], errors='coerce')
            return numeric_col
        
        def calculate_auc_binary(df, group_col, group1, group2, biomarker_col):
            """Calculate AUC for binary classification"""
            # Filter data for the two groups
            group1_data = df[df[group_col] == group1]
            group2_data = df[df[group_col] == group2]
            
            # Get biomarker values
            group1_values = clean_numeric_column(group1_data, biomarker_col).dropna()
            group2_values = clean_numeric_column(group2_data, biomarker_col).dropna()
            
            if len(group1_values) == 0 or len(group2_values) == 0:
                print(f"No valid data found for {biomarker_col}")
                return None
            
            # Create labels (1 for group1, 0 for group2)
            y_true = [1] * len(group1_values) + [0] * len(group2_values)
            y_scores = list(group1_values) + list(group2_values)
            
            # Calculate AUC
            auc = roc_auc_score(y_true, y_scores)
            return auc
        
        # Add helper functions to globals
        self.globals_dict['clean_numeric_column'] = clean_numeric_column
        self.globals_dict['calculate_auc_binary'] = calculate_auc_binary
        
        # If no file path provided, ask user to upload
        if not self.file_path:
            self.file_path = self.get_file_from_user()
        
        if self.file_path:
            self.load_dataset()
    
    def get_file_from_user(self):
        """Get file path from user - either through dialog or manual input"""
        print("🔬 Welcome to Interactive Data Analyzer!")
        print("=" * 50)
        print("📁 Please provide your dataset file:")
        print("   1. Type file path manually")
        print("   2. Use file dialog (requires tkinter)")
        print("=" * 50)
        
        while True:
            choice = input("Choose option (1 or 2): ").strip()
            
            if choice == '1':
                file_path = input("📁 Enter full file path (CSV/Excel): ").strip()
                if file_path and (file_path.endswith(('.csv', '.xlsx', '.xls'))):
                    return file_path
                else:
                    print("❌ Please provide a valid CSV or Excel file path")
                    
            elif choice == '2':
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    
                    root = tk.Tk()
                    root.withdraw()  # Hide the main window
                    
                    file_path = filedialog.askopenfilename(
                        title="Select Dataset File",
                        filetypes=[
                            ("Excel files", "*.xlsx *.xls"),
                            ("CSV files", "*.csv"),
                            ("All files", "*.*")
                        ]
                    )
                    root.destroy()
                    
                    if file_path:
                        return file_path
                    else:
                        print("❌ No file selected")
                        
                except ImportError:
                    print("❌ tkinter not available. Please use option 1 instead.")
                    
            else:
                print("❌ Please choose 1 or 2")
    
    def upload_new_dataset(self):
        """Allow user to upload a new dataset during chat"""
        print("📁 Uploading new dataset...")
        new_file_path = self.get_file_from_user()
        
        if new_file_path:
            self.file_path = new_file_path
            self.df = None  # Clear current dataset
            self.conversation_history = []  # Clear history
            self.load_dataset()
            if self.df is not None:
                print("✅ New dataset loaded successfully!")
                return True
        
        print("❌ Failed to load new dataset")
        return False
        
    def load_dataset(self):
        """Load the dataset once at startup"""
        try:
            if self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
                self.df = pd.read_excel(self.file_path)
            elif self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            else:
                raise ValueError("File must be .xlsx, .xls, or .csv")
            
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            print(f"📋 Columns: {list(self.df.columns)}")
            print(f"🧬 Data types: {dict(self.df.dtypes)}")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            self.df = None
    
    def chat_with_ollama(self, prompt):
        """Send prompt to Ollama"""
        url = "http://localhost:11434/api/generate"
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            return result["response"]
        except Exception as e:
            return f"Error connecting to Ollama: {e}"
    
    def extract_python_code(self, text):
        """Extract Python code blocks from Ollama response"""
        # Updated patterns to properly handle language identifiers
        patterns = [
            r'```python\s*\n(.*?)\n```',     # ```python blocks
            r'```py\s*\n(.*?)\n```',         # ```py blocks  
            r'```\s*\n(.*?)\n```',           # ``` blocks without language
            r'```python(.*?)```',            # inline ```python blocks
            r'```(.*?)```'                   # fallback for any ``` blocks
        ]
        
        code_blocks = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                # Clean up the code block
                code_block = match.strip()
                
                # Skip if it's just a language identifier
                if code_block.lower() in ['python', 'py', '']:
                    continue
                    
                # Remove any remaining language identifiers at the start
                lines = code_block.split('\n')
                if lines and lines[0].strip().lower() in ['python', 'py']:
                    code_block = '\n'.join(lines[1:]).strip()
                
                if code_block:
                    code_blocks.append(code_block)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_blocks = []
        for block in code_blocks:
            block_clean = block.strip()
            if block_clean and block_clean not in seen:
                seen.add(block_clean)
                unique_blocks.append(block_clean)
        
        return unique_blocks
    
    def get_dataset_context(self):
        """Get current dataset context for prompts"""
        if self.df is None:
            return "No dataset loaded."
        
        # Get sample values for key columns
        tumor_types = self.df['Tumor_type'].value_counts().head()
        
        return f"""
Current Dataset Context:
- Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns
- Columns: {list(self.df.columns)}
- Data types: {dict(self.df.dtypes)}
- Missing values: {dict(self.df.isnull().sum())}

Key Information:
- Tumor types: {dict(tumor_types)}
- Many biomarker columns are stored as 'object' type but contain numeric values
- Use clean_numeric_column(df, 'column_name') to convert columns to numeric
- Use calculate_auc_binary(df, 'Tumor type', 'group1', 'group2', 'biomarker') for AUC analysis

Sample data (first 3 rows):
{self.df.head(3).to_string()}

Available helper functions:
- clean_numeric_column(df, col_name): Convert column to numeric
- calculate_auc_binary(df, group_col, group1, group2, biomarker_col): Calculate AUC for binary classification

Recent conversation history:
{self.get_recent_history()}
"""
    
    def get_recent_history(self):
        """Get recent conversation for context"""
        if len(self.conversation_history) > 3:
            return "\n".join(self.conversation_history[-3:])
        return "\n".join(self.conversation_history)
    
    def process_question(self, question):
        """Process user question and generate analysis"""
        if self.df is None:
            return "❌ No dataset loaded. Please check the file path."
        
        # Add to conversation history
        self.conversation_history.append(f"User: {question}")
        
        # Create comprehensive prompt
        dataset_context = self.get_dataset_context()
        
        prompt = f"""{dataset_context}

User Question: {question}

IMPORTANT INSTRUCTIONS:
- The dataset is ALREADY LOADED as 'df' - DO NOT try to load or read any files
- The dataframe 'df' contains all the data shown above
- Generate Python code that works directly with the existing 'df' variable
- All required libraries are already imported (pandas as pd, numpy as np, etc.)

Please analyze this question and provide ONLY executable Python code in code blocks. 

Requirements:
1. Use the existing 'df' variable (do NOT load data from files)
2. Generate complete, executable Python code
3. Include clear print statements for results
4. Create visualizations if appropriate
5. Add comments explaining the analysis

Example format:
```python
# Analysis of the question
print("Results:")
# Your code here using df
```

Generate code to answer: {question}
"""
        
        print("🤖 Analyzing your question...")
        response = self.chat_with_ollama(prompt)
        
        # Extract and execute code
        code_blocks = self.extract_python_code(response)
        
        if not code_blocks:
            print("🤖 No executable code generated. Here's the full response:")
            print("-" * 60)
            print(response)
            print("-" * 60)
            print("💡 Try rephrasing your question or being more specific.")
            return response
        
        print("📝 Generated Analysis Code:")
        print("=" * 60)
        
        # Execute code blocks
        results = []
        
        # Update globals with current df and ensure data types are handled
        self.globals_dict['df'] = self.df.copy()  # Use a copy to avoid modifications
        
        for i, code in enumerate(code_blocks):
            print(f"\n🚀 Executing Analysis Block {i+1}:")
            print("-" * 40)
            print(code)
            print("-" * 40)
            
            try:
                # Execute with proper global and local namespace
                exec(code, self.globals_dict)
                results.append(f"Block {i+1}: Executed successfully")
                print(f"✅ Block {i+1} completed successfully")
                
            except Exception as e:
                error_msg = f"❌ Error in block {i+1}: {str(e)}"
                print(error_msg)
                results.append(error_msg)
                
                # Try to provide helpful debugging info
                if "not defined" in str(e):
                    print(f"💡 Tip: Variable '{str(e).split()[1].strip(chr(39))}' not found. Available variables: df, pd, np, plt, sns")
                elif "invalid literal" in str(e):
                    print("💡 Tip: There might be data type issues. Check for non-numeric values in numeric columns.")
                elif "KeyError" in str(e):
                    print("💡 Tip: Column name not found. Check column names with df.columns")
        
        # Add results to conversation history
        self.conversation_history.append(f"Analysis completed: {'; '.join(results)}")
        
        return results
    
    def show_help(self):
        """Show available commands and examples"""
        help_text = """
🔬 Interactive Data Analysis Commands:

📊 ANALYSIS QUESTIONS (examples):
• "What are the top 5 most important biomarkers?"
• "Create ROC curves for colorectal cancer detection"
• "Perform t-tests comparing cancer vs control groups"
• "Show correlation heatmap of all variables"
• "Build a machine learning model to predict cancer"
• "Find outliers in the protein data"
• "Calculate AUC scores for each biomarker"
• "Perform PCA analysis on the dataset"

💬 CHAT COMMANDS:
• help - Show this help message
• info - Show dataset information
• upload - Upload a new dataset file
• history - Show conversation history
• clear - Clear conversation history
• quit/exit - Exit the analyzer

🧬 BIOMARKER-SPECIFIC EXAMPLES:
• "Which proteins show the highest fold change?"
• "Statistical significance testing for each marker"
• "Feature selection for early cancer detection"
• "Compare biomarker performance using ROC analysis"

Just type your question naturally - I'll generate and run the analysis code!
"""
        print(help_text)
    
    def show_info(self):
        """Show current dataset information"""
        if self.df is None:
            print("❌ No dataset loaded")
            return
        
        print("📊 Current Dataset Information:")
        print(f"📁 File: {self.file_path}")
        print(f"📐 Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"📋 Columns: {list(self.df.columns)}")
        print(f"🧮 Data Types:")
        for col, dtype in self.df.dtypes.items():
            print(f"   {col}: {dtype}")
        print(f"❓ Missing Values: {self.df.isnull().sum().sum()} total")
        
        print("\n📈 Quick Statistics:")
        print(self.df.describe())
    
    def start_chat(self):
        """Start the interactive chat session"""
        print("🔬 Interactive Data Analysis Chat")
        print("=" * 50)
        print("Ask me anything about your dataset!")
        print("Type 'help' for examples, 'upload' to load new file, 'quit' to exit")
        print("=" * 50)
        
        if self.df is None:
            print("❌ No dataset loaded. Type 'upload' to load a dataset.")
        
        while True:
            try:
                user_input = input("\n🧑‍🔬 Your Question: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Happy analyzing!")
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                elif user_input.lower() == 'info':
                    self.show_info()
                elif user_input.lower() == 'upload':
                    self.upload_new_dataset()
                elif user_input.lower() == 'history':
                    print("\n📜 Conversation History:")
                    for item in self.conversation_history:
                        print(f"  {item}")
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("🧹 Conversation history cleared!")
                else:
                    # Check if dataset is loaded before processing questions
                    if self.df is None:
                        print("❌ No dataset loaded. Please type 'upload' to load a dataset first.")
                        continue
                    
                    # Process analysis question
                    print("\n" + "="*60)
                    self.process_question(user_input)
                    print("="*60)
                    
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

# Quick start function
def start_interactive_analysis(file_path=None):
    """Quick way to start interactive analysis"""
    analyzer = InteractiveDataAnalyzer(file_path)
    analyzer.start_chat()

if __name__ == "__main__":
    # Start interactive mode - will prompt user for file
    print("🚀 Starting Interactive Data Analyzer...")
    start_interactive_analysis()
