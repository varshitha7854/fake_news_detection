# Fake News Detector

Simple Python project that trains a RandomForest text classifier using TF‑IDF features to detect fake news.

## Project structure
- `main.py` — training and evaluation script
- `news.csv` — dataset (must be added by you)
- `.venv/` — optional virtual environment (should be ignored)
- `.gitignore` — recommended to exclude `.venv`, `__pycache__`, etc.

## Requirements
- Python 3.8+
- pandas
- scikit-learn

Example requirements (optional):
```
pandas
scikit-learn
```

## Setup (Windows, PowerShell)
1. Open project folder:
   ```powershell
   cd D:\Varshithamuthineni.ai\fake-news-detector
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate
   ```

3. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install pandas scikit-learn
   ```

## Dataset
Place `news.csv` in the project root (same folder as `main.py`) or update `CSV_PATH` inside `main.py` to the full path.

Required columns:
- `text` — article content (string)
- `label` — target label (e.g., `fake` / `real`)

Example CSV rows:
```
"text","label"
"The government passed a new law...", "real"
"Scientists discover immortality pill...", "fake"
```

## Run
From the project directory:
```powershell
python main.py
```
If `news.csv` is not in the project folder, update `CSV_PATH` in `main.py` to an absolute path, for example:
```python
CSV_PATH = r"D:\Varshithamuthineni.ai\fake-news-detector\news.csv"
```

## What the script does
1. Loads `news.csv`.
2. Validates presence of `text` and `label` columns.
3. Converts text to TF‑IDF vectors.
4. Encodes labels and splits data (80/20).
5. Trains a `RandomForestClassifier`.
6. Prints test accuracy and label mapping.

## Troubleshooting
- ModuleNotFoundError: install packages with:
  ```powershell
  python -m pip install pandas scikit-learn
  ```
- FileNotFoundError for `news.csv`: ensure file exists at `CSV_PATH` or update the path.
- Pylance/IDE warnings: ensure VS Code uses the same Python interpreter (use "Python: Select Interpreter").

## Git
Add a `.gitignore` to avoid committing `.venv` and large data:
```text
# .gitignore
.venv/
__pycache__/
*.pyc
.vscode/
news.csv    # optional: if you don't want to push dataset
```

## Extending
- Add CLI args or an env var to pass CSV path.
- Save trained model with `joblib`.
- Add a test set or cross-validation and more metrics (precision/recall/F1).
- Create a web API for inference (Flask/FastAPI).

## License
MIT License — see LICENSE file or add your preferred license.

## Author / Contact
Repository: https://github.com/varshitha7854/fake_news_detection
