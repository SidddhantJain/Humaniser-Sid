# 🤖➡️👨 AI Text Humanizer

An advanced tool to transform robotic, AI-generated text into natural, human-like writing that can bypass AI detection tools.

## 🚀 Features

- **Multiple AI Models**: Uses T5 and Pegasus models for diverse paraphrasing
- **Advanced Techniques**: Vocabulary diversification, sentence restructuring, natural flow enhancement
- **Batch Processing**: Handle multiple texts and files at once
- **Academic Focus**: Preserves academic tone while making text more natural
- **Undetectable Output**: Creates human-like text that passes AI detection tools
- **Multiple Interfaces**: Simple, advanced, and batch processing versions

## 📁 Files

1. **`humanizer_app.py`** - Advanced version with multiple models and sophisticated techniques
2. **`humanizer_simple.py`** - Simplified version with reliable single model
3. **`humanizer_batch.py`** - Batch processing version for files and multiple texts

## 🛠️ Installation

### Prerequisites

1. Python 3.8+ installed
2. Virtual environment (recommended)

### Setup

```bash
# Clone or download the project
cd Humanizer

# Create virtual environment (if not already created)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install required packages
pip install gradio transformers torch tiktoken nltk textstat protobuf pandas

# Run the application
python humanizer_app.py        # Advanced version
# OR
python humanizer_simple.py     # Simple version  
# OR
python humanizer_batch.py      # Batch processing version
```

## 🎯 Usage

### Basic Usage

1. Run one of the Python files
2. Open your browser to the displayed URL (usually http://127.0.0.1:7860)
3. Paste your AI-generated text
4. Select humanization level
5. Click "Humanize" and get natural, human-like output

### Humanization Levels

- **Light**: Basic paraphrasing with minimal changes
- **Moderate/Medium**: Paraphrasing + vocabulary variations + natural connectors
- **Heavy**: All techniques + sentence structure modifications + advanced variations

### Batch Processing

The batch processor (`humanizer_batch.py`) supports:
- **.txt files**: Processes paragraph by paragraph
- **.csv files**: Adds a 'humanized' column with processed text

## 🔧 How It Works

### Advanced Techniques Used

1. **Multi-Model Paraphrasing**: Uses multiple AI models to avoid patterns
2. **Vocabulary Diversification**: Replaces words with contextual synonyms
3. **Sentence Structure Variation**: Modifies sentence patterns for natural flow
4. **Academic Connector Integration**: Adds natural transitional phrases
5. **Hedging Language**: Incorporates academic hedging for natural tone
6. **Smart Chunking**: Processes long texts in optimal chunks

### AI Models Used

- **T5 Paraphrase (Primary)**: `Vamsi/T5_Paraphrase_Paws`
- **Pegasus (Secondary)**: `tuner007/pegasus_paraphrase`
- **NLTK WordNet**: For synonym replacement
- **Custom Algorithms**: For structure and flow optimization

## 📊 Example Transformations

### Input (AI-generated):
```
The implementation of machine learning algorithms in data processing systems demonstrates significant improvements in efficiency and accuracy metrics across various benchmark datasets.
```

### Output (Humanized):
```
Implementing machine learning algorithms within data processing frameworks shows notable enhancements in both efficiency and accuracy measures when evaluated across different benchmark datasets. These improvements suggest that such approaches can effectively optimize computational performance.
```

## 🎮 Advanced Features

### Multi-Level Processing
- Processes texts of any length by intelligent chunking
- Maintains context across chunks
- Preserves academic integrity

### Natural Variations
- Dynamic vocabulary replacement
- Contextual synonym selection  
- Academic phrase integration
- Sentence flow optimization

### Error Handling
- Graceful fallbacks if models fail
- Multiple backup techniques
- Robust error recovery

## 🔍 Best Practices

1. **Input Quality**: Use complete sentences and proper grammar
2. **Length Considerations**: Works best with 50-1000 word chunks
3. **Context Preservation**: Review output to ensure meaning is maintained
4. **Multiple Passes**: For heavy humanization, consider multiple rounds
5. **Manual Review**: Always review output for accuracy and flow

## 🚫 Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure protobuf is installed: `pip install protobuf`
   - Check internet connection for model downloads
   - Try the simple version if advanced fails

2. **Memory Issues**:
   - Reduce text chunk size
   - Use lighter humanization levels
   - Close other applications

3. **Performance Issues**:
   - Use GPU if available
   - Process smaller texts
   - Try the simple version

## ⚖️ Ethical Usage

This tool is designed for:
- ✅ Improving writing quality
- ✅ Learning natural language patterns  
- ✅ Enhancing academic writing
- ✅ Content optimization

Please use responsibly and:
- 🚫 Don't use for plagiarism
- 🚫 Don't violate academic integrity policies
- 🚫 Don't misrepresent authorship
- 🚫 Don't use for deceptive purposes

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest improvements
- Add new models
- Enhance techniques

## 📄 License

This project is for educational and research purposes. Please respect academic integrity and use responsibly.

---

**Made with ❤️ for better academic writing**
