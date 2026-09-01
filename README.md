# Learning the Cue or Learning the Word?

Code for Learning the Cue or Learning the Word? Analyzing Generalization in Metaphor Detection for Verbs

**Abstract**
Metaphor detection models achieve strong benchmark performance, yet it remains unclear whether this reflects transferable generalization or lexical memorization. To address this, we analyze generalization in metaphor detection through RoBERTa, the shared backbone of many state-of-the-art systems, focusing on English verbs using the VU Amsterdam Metaphor Corpus. We introduce a controlled lexical hold-out setup where all instances of selected target lemmas are strictly excluded from fine-tuning, and compare predictions on these Held-out lemmas against Exposed lemmas (verbs seen during fine-tuning). While the model performs
best on Exposed lemmas, it maintains robust performance on Held-out lemmas. Further analysis reveals that sentence context alone is sufficient to match full-model performance on Held-out lemmas, whereas static verb-level embeddings are not. Together, these results suggest that generalization is primarily driven by "learning the cue" (transferable contextual patterns), while "learning the word" (verb-specific memorization) provides an additive boost when lexical exposure is available.

## Setup

Clone the repository:

```bash
git clone https://github.com/seinan9/learning-the-cue.git
cd learning-the-cue
```

Install the dependencies:

**With [uv](https://docs.astral.sh/uv/) (recommended)**

```bash
uv sync
source .venv/bin/activate
```

**With plain pip**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> On Windows, activate with `.venv\Scripts\activate` instead.

## Repository Structure

```
learning-the-cue
├── data              <- VUA corpus (train/test CSVs)
├── learning_the_cue  <- shared components
├── pyproject.toml
├── README.md
├── scripts.          <- experiment scripts, one subfolder per pipeline stage
└── uv.lock
```
