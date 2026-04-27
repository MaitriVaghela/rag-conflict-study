import os

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate



# training data
# temperature=0 so the model gives consistent, deterministic answers.
questions = [{
    "id": "privilgeDetermination_001",
    "question": "Can a user with 7 purchases on their account combine shipments?",
    "doc_correct": "Users must have completed at least 10 purchases to be eligible for combined shipping.",
    "doc_wrong": "Users must have completed at least 5 purchases to be eligible for combined shipping.",
    "correct_answer": "No"
}]


# setting up model and prompt
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""Answer the question using only the two documents below.
Give a direct Yes or No answer, followed by one sentence explanation.

Document 1: {doc1}
Document 2: {doc2}

Question: {question}
Answer:""")

# print("testing134")

results = []

for q in questions:
    print(f"Running test: {q['id']}")

    # first the prompt fills in the placeholders, then the model generates a response.
    r_correct_first = (prompt | llm).invoke({
        "doc1": q["doc_correct"],
        "doc2": q["doc_wrong"],
        "question": q["question"]
        })
    
    # experiment with the order of the documents to see if it affects the model's answer.
    r_wrong_first = (prompt | llm).invoke({
        "doc1": q["doc_wrong"],
        "doc2": q["doc_correct"],
        "question": q["question"]
        })
    
    results.append({
        "id": q["id"],
        "question": q["question"],
        "correct_answer": q["correct_answer"],
        "response_correct_first": r_correct_first.content,
        "response_wrong_first": r_wrong_first.content,
        "correct_first_is_correct": "",
        "wrong_first_is_correct": ""
    })

    print(f"Done: {q['id']}")

df = pd.DataFrame(results)
df.to_csv("experiment_results.csv", index=False)
print("Experiment completed. Results saved to experiment_results.csv")


