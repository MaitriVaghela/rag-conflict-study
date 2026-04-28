import os
import json
import subprocess
import sys

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

with open("online_market_privilege_determination_questions.json") as f:
    questions = json.load(f)


# setting up model and prompt
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""Answer the question using only the two documents below.
Give a direct Yes or No answer, followed by one sentence explanation.

Document 1: {doc1}
Document 2: {doc2}

Question: {question}
Answer:""")

def is_correct(response, correct_answer):
    return response.strip().lower().startswith(correct_answer.strip().lower())

results = []

# experiment with the order of the documents to see if it affects the model's answer.
for q in questions:
    print(f"Running test: {q['id']}")

    r_correct_first = (prompt | llm).invoke({
        "doc1": q["doc_correct"],
        "doc2": q["doc_wrong"],
        "question": q["question"]
        })

    r_wrong_first = (prompt | llm).invoke({
        "doc1": q["doc_wrong"],
        "doc2": q["doc_correct"],
        "question": q["question"]
        })

    results.append({
        "id": q["id"],
        "question": q["question"],
        "correct_answer": q["correct_answer"],
        "model_response_correct_first": r_correct_first.content,
        "model_response_wrong_first": r_wrong_first.content,
        "correct_doc_first_accuracy": is_correct(r_correct_first.content, q["correct_answer"]),
        "wrong_doc_first_accuracy": is_correct(r_wrong_first.content, q["correct_answer"])
    })

    print(f"Done: {q['id']}")

df = pd.DataFrame(results)

df.to_csv("online_market_privilege_determination_experiment_results.csv", index=False)

# execute the analysis notebook to generate results and visualizations based on the experiment results.
subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace", "online_market_privilege_determination_analysis.ipynb"])

print("Experiment completed. Results saved to online_market_privilege_determination_experiment_results.csv")




