# AutoTag - Support Ticket Classification

A Python script that uses Groq's LLM API to automatically classify support tickets into predefined categories using both zero-shot and few-shot learning approaches.

## Features

- **Zero-shot Classification**: Classifies tickets without training examples
- **Few-shot Classification**: Uses examples to improve classification accuracy
- **Accuracy Evaluation**: Calculates Top-1 and Top-3 accuracy metrics
- **Batch Processing**: Handles large datasets efficiently with progress tracking
- **Multiple Output Formats**: Saves results in CSV format with evaluation metrics

## Prerequisites

- Python 3.7+
- Groq API key
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone or download the script files
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project directory with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

### Basic Usage

1. Place your support tickets CSV file in the project directory
2. Ensure your CSV has a "Ticket Description" column
3. Run the script:
   ```bash
   python TicketTagger.py
   ```

### Input CSV Format

Your CSV file should contain:
- `Ticket Description`: The text content of each support ticket
- `Ticket Type`: The actual/true category (for evaluation purposes)

Example:
```csv
Ticket Description,Ticket Type
"I can't log into my account",Technical issue
"How do I cancel my subscription?",Cancellation request
"I was charged twice",Billing inquiry
```

### Output Files

The script generates several output files:

1. **`classified_tickets_eval_zshot.csv`**: Zero-shot classification results with accuracy metrics
2. **`classified_tickets_few_shot_eval.csv`**: Few-shot classification results with accuracy metrics

### Output Columns

- `Ticket Description`: Original ticket text
- `Ticket Type`: True category (from input)
- `Predicted Tags`: Model predictions (zero-shot)
- `Predicted Tags (Few-shot)`: Model predictions (few-shot)
- `Top-1 Match`: Whether the top prediction matches the true category
- `Top-3 Match`: Whether the true category appears in top 3 predictions
- `Top-1 Match (FS)`: Same as above but for few-shot results
- `Top-3 Match (FS)`: Same as above but for few-shot results

## Configuration

### Tags

Modify the `TAGS` list in the script to match your use case:

```python
TAGS = [
    'Technical issue', 'Billing inquiry', 'Cancellation request',
    'Product inquiry', 'Refund request'
]
```

### Model Selection

Change the model by modifying the `model` parameter:

```python
# Current default
model="llama-3.1-8b-instant"

# Other available models
model="meta-llama/llama-4-scout-17b-16e-instruct"
```

### Sample Size

Adjust the number of tickets to process by changing the `sample(n=25)` parameter:

```python
df = df[df["Ticket Description"].notnull()].sample(n=25, random_state=42).copy()
```

## Evaluation Metrics

The script calculates two accuracy metrics:

- **Top-1 Accuracy**: Percentage of cases where the model's top prediction matches the true category
- **Top-3 Accuracy**: Percentage of cases where the true category appears in the model's top 3 predictions

## API Usage

The script uses Groq's API for inference. Key parameters:

- **Temperature**: 0.3 (controls randomness)
- **Max Tokens**: 150 (limits response length)
- **Streaming**: Disabled for batch processing

## Error Handling

- Invalid API keys or network issues are caught and logged
- Missing CSV columns trigger clear error messages
- Malformed responses are handled gracefully

## Troubleshooting

### Common Issues

1. **"Ticket Description column not found"**
   - Check your CSV column names
   - Ensure exact spelling and case

2. **API errors**
   - Verify your Groq API key in `.env`
   - Check your internet connection
   - Ensure you have sufficient API credits

3. **Empty predictions**
   - Check if your ticket descriptions are meaningful
   - Verify the model name is correct

### Performance Tips

- Reduce sample size for faster testing
- Use smaller models for quicker inference
- Process in smaller batches for large datasets

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Contributing

Improvements and bug fixes are welcome! Please ensure your changes maintain the existing functionality and add appropriate error handling. 