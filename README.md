# ConflictRAG - LLM Reliability Under Conflicting Documents

## Overview

This project investigates whether a language model can reliably determine user privileges when provided with conflicting policy documents simulating a real-world problem where rules governing user entitlements are spread across multiple sources that do not fully agree with each other.

The core question: **does the model give consistent, correct answers when two documents describe the same rule differently?**

## Motivation

In many platforms and applications, the rules governing what users are allowed to do are spread across multiple sources like policy documents, configuration files,  wikis, and system documentation. These sources are often maintained independently and may not stay in sync, leading to situations where different documents describe the same rule differently.

When a language model retrieves information from multiple such sources, it may encounter conflicting thresholds or conditions without a clear signal about which document is correct. This creates a reliability problem: the model's answer may change depending on which document it sees first, even when the underlying question is the same.

This experiment measures that sensitivity directly.

## Experiment Design

Each test case consists of:
- A **question** about whether a user is entitled to perform an action
- A **correct document** containing the accurate rule
- A **wrong document** containing a plausible but incorrect version of the same rule
- A **correct answer** (Yes or No)

The model is run twice per question:
1. **Correct doc first**: correct document is Document 1, wrong document is Document 2
2. **Wrong doc first**: wrong document is Document 1, correct document is Document 2

Accuracy is measured by checking whether the model's response starts with the correct Yes/No answer.

## Key Finding

The model exhibited **recency bias**. It tended to follow whichever document appeared last rather than whichever was more authoritative:

- Correct doc first → **73% accuracy**
- Wrong doc first → **87% accuracy**

This means that in a RAG system where document retrieval order is not controlled, the same user privilege question can produce different answers depending on how results are ordered. It is a meaningful reliability risk in any system where decisions have real consequences.

## Project Structure

```
├── online_market_privilege_determination_questions.json   # Test dataset (15 questions)
├── online_market_privilege_determination_experiment.py    # Runs the experiment, saves results
├── online_market_privilege_determination_analysis.ipynb   # Analysis and failure breakdown
├── online_market_privilege_determination_experiment_results.csv  # Output (generated)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install langchain-openai langchain-core pandas nbconvert python-dotenv
```

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_key_here
```

## Running the Experiment

```bash
python online_market_privilege_determination_experiment.py
```

This will:
1. Load the questions from the JSON file
2. Run each question twice (correct doc first, wrong doc first) using `gpt-4o-mini`
3. Save results to the CSV file
4. Automatically execute the analysis notebook and save outputs into it

Open `online_market_privilege_determination_analysis.ipynb` to view the full breakdown including which questions failed and under which ordering.
