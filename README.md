# CNN/DailyMail Text Summarization (Abstractive)

This project applies **pre-trained encoder–decoder models** to generate concise summaries from long news articles. By combining careful text preprocessing, transformer-based sequence-to-sequence models (e.g., **T5**, **BART**, optionally **Pegasus**), and **ROUGE**-based evaluation, the goal is to build reliable **abstractive** summaries that retain core facts while compressing the original text.

---

## Dataset

We use the **CNN/DailyMail** news dataset (available via Hugging Face Datasets as `cnn_dailymail` or on Kaggle).

* Each example contains:
  * **Article** — long-form news text
  * **Highlights** — human-written reference summary
* Widely used as a benchmark for **abstractive summarization**.

> Load with: `datasets.load_dataset('cnn_dailymail', '3.0.0')` or download from Kaggle and point the notebook to the local files.

---

## Project Workflow

The notebook is structured as follows:

### 1. Data Loading & Exploration

* Load the CNN/DailyMail dataset into a pandas DataFrame (or directly via `datasets`).
* Inspect samples, length stats (article vs. summary), and basic distributions.
* (Optional) Visualize length histograms to pick truncation limits.

### 2. Text Preprocessing

Light normalization before tokenization:
* Trim/normalize whitespace and unusual characters.
* (Optional) Lowercasing — model/tokenizer dependent (many models are case-sensitive).
* Filter very short/malformed samples.
* Ensure summaries are not empty and within max length.

> Heavy cleaning (e.g., stopword removal, lemmatization) is **not required** for transformer tokenizers.

### 3. Modeling — Encoder–Decoder

We experiment with state-of-the-art abstractive models:

* **T5** (e.g., `t5-small`, `t5-base`) — prefix task with `summarize:`
* **BART** (e.g., `facebook/bart-base`, `facebook/bart-large-cnn`)
* *(Optional)* **Pegasus** (e.g., `google/pegasus-cnn_dailymail`)

Key settings:
* Max input tokens (truncation) and max summary tokens.
* Training hyperparameters: batch size, learning rate, epochs, warmup steps.
* Generation settings: beam search, length penalty, no-repeat n-gram size.

### 4. Training / Fine-tuning

* Tokenize (article → input_ids; highlights → labels).
* Fine-tune with Hugging Face **Transformers** `Trainer` or PyTorch loops.
* Save the best checkpoint (monitor validation ROUGE or loss).

### 5. Evaluation (ROUGE)

* Compute **ROUGE-1/2/L** between generated summaries and references.
* Report scores on the validation/test split.
* (Optional) Log examples of good/weak generations for qualitative review.

### 6. Inference & Batch Summarization

* Run inference on custom articles or files.
* Export generated summaries to CSV/JSON for inspection.
* (Optional) Add a minimal **Streamlit** UI to paste text and view the summary.

### 7. (Bonus) Extractive Baseline

To complement abstractive models:
* Implement **TextRank** (e.g., `sumy` or custom) or **Gensim** summarization.
* Compare ROUGE/qualitative differences vs. T5/BART.

---

## Results

* The notebook reports **ROUGE-1/2/L** on the held-out split and prints sample summaries for manual inspection.
* Qualitatively, abstractive models produce fluent, paraphrased summaries that align with the main points of the source article.
* Please update this section with your final scores after running the notebook on your environment.

**Example (template — replace with your scores):**
```
Model: facebook/bart-large-cnn
ROUGE-1: XX.X
ROUGE-2: YY.Y
ROUGE-L: ZZ.Z
```

---

## Tech Stack

* **Python**
* **Pandas**, **NumPy** — data handling
* **Hugging Face Transformers**, **Tokenizers**
* **Datasets** (Hugging Face) or **Kaggle** files
* **ROUGE** (e.g., `evaluate` / `rouge-score`)
* **Matplotlib / Seaborn** — optional plots

---

## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # or minimal set:
   pip install transformers datasets evaluate rouge-score pandas torch
   ```

2. **Get the dataset**
   * **Hugging Face:**
     ```python
     from datasets import load_dataset
     ds = load_dataset('cnn_dailymail', '3.0.0')
     ```
   * **Kaggle:**
     Download CNN/DailyMail and set the notebook paths accordingly.

3. **Open the notebook**
   * `text-summarization.ipynb`
   * Run cells in order (train → evaluate → infer).

4. **(Optional) Inference only**
   * Load a pre-trained checkpoint in the notebook and run the inference cell on your own text.

---

## Repository Structure (suggested)

```
.
├── text-summarization.ipynb     # main notebook (training/eval/inference)
├── requirements.txt             # dependencies
├── data/                        # optional local storage for Kaggle data
├── outputs/                     # generated summaries, metrics, checkpoints
└── README.md                    # this file
```

---

## Key Takeaways

* **Abstractive summarization** with encoder–decoder models (T5/BART) produces fluent summaries and is the current standard for news datasets like CNN/DailyMail.
* **Proper truncation** and generation settings (beams, length penalty) significantly impact quality.
* **ROUGE** remains the common quantitative metric; qualitative review is still important.
* An **extractive baseline** (TextRank/Gensim) provides a useful point of comparison for speed and faithfulness.

---

## Acknowledgments

* Hugging Face **Transformers**, **Datasets**, and **Evaluate** libraries
* CNN/DailyMail dataset creators and maintainers
* Open-source contributors whose models/checkpoints are used in this project
