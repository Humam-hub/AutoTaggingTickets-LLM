import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load Groq API key from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# Define tags (adjust based on your dataset or use unique ticket types)
TAGS = [
    'Technical issue', 'Billing inquiry', 'Cancellation request',
    'Product inquiry', 'Refund request'
]

# Prompt builder
def build_zero_shot_prompt(ticket_text: str) -> str:
    return f"""
You are an intelligent support assistant. Read the support ticket below and classify it into the 3 most relevant tags from the following list:

Tags: {", ".join(TAGS)}

Ticket Description:
\"\"\"
{ticket_text}
\"\"\"

Respond only with a Python list of the 3 most relevant tags.
"""

# Function to get classification from Groq LLM
def classify_ticket_zero_shot(ticket_text: str, model="llama-3.1-8b-instant"):
    prompt = build_zero_shot_prompt(ticket_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=150,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return f"Error: {e}"

# Load dataset
df = pd.read_csv("/content/support_tickets.csv")

# Filter tickets with non-empty descriptions
df = df[df["Ticket Description"].notnull()].sample(n=25, random_state=42).copy()

# Apply zero-shot classifier
tqdm.pandas(desc="Classifying tickets")
df["Predicted Tags"] = df["Ticket Description"].progress_apply(classify_ticket_zero_shot)

def top1_accuracy(predicted_tags, true_tag):
    try:
        predicted_list = eval(predicted_tags)  # turns string to list
        return int(predicted_list[0].lower() == true_tag.lower())
    except:
        return 0

def top3_accuracy(predicted_tags, true_tag):
    try:
        predicted_list = eval(predicted_tags)
        return int(true_tag.lower() in [tag.lower() for tag in predicted_list])
    except:
        return 0

df["Top-1 Match"] = df.apply(lambda row: top1_accuracy(row["Predicted Tags"], row["Ticket Type"]), axis=1)
df["Top-3 Match"] = df.apply(lambda row: top3_accuracy(row["Predicted Tags"], row["Ticket Type"]), axis=1)

# Accuracy scores
top1_score = df["Top-1 Match"].mean()
top3_score = df["Top-3 Match"].mean()

print(f"\n🔍 Top-1 Accuracy: {top1_score:.2%}")
print(f"🔍 Top-3 Accuracy: {top3_score:.2%}")

df[["Ticket Description", "Ticket Type", "Predicted Tags", "Top-1 Match", "Top-3 Match"]].to_csv("classified_tickets_eval_zshot.csv", index=False)

# Collect 5 examples per tag from dataset
few_shot_examples = []
for tag in TAGS:
    examples = df[df['Ticket Type'] == tag].head(5)
    for _, row in examples.iterrows():
        few_shot_examples.append({
            "description": row["Ticket Description"],
            "tags": [tag]
        })

# Few-shot prompt builder
def build_few_shot_prompt(ticket_text: str) -> str:
    prompt = "You are a helpful support assistant. Based on the examples below, classify the following ticket into the 3 most relevant tags from this list:\n\n"
    prompt += f"Tags: {', '.join(TAGS)}\n\n"
    prompt += f"Respond only with a Python list of the 3 most relevant tags.\n\n"

    for i, ex in enumerate(few_shot_examples):
        prompt += f"Example {i+1}:\n"
        prompt += f'Ticket: """{ex["description"]}"""\n'
        prompt += f'Tags: {ex["tags"]}\n\n'

    prompt += f'Ticket: """{ticket_text}"""\n'
    prompt += "Tags:"
    return prompt

# Classification function
def classify_ticket_few_shot(ticket_text: str, model="llama-3.1-8b-instant"):
    prompt = build_few_shot_prompt(ticket_text)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=150,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return f"Error: {e}"

# Sample 25 for evaluation
df_eval = df.sample(n=25, random_state=42).copy()

# Predict using few-shot
tqdm.pandas(desc="Few-shot classifying")
df_eval["Predicted Tags (Few-shot)"] = df_eval["Ticket Description"].progress_apply(classify_ticket_few_shot)

# Evaluate
df_eval["Top-1 Match (FS)"] = df_eval.apply(lambda row: top1_accuracy(row["Predicted Tags (Few-shot)"], row["Ticket Type"]), axis=1)
df_eval["Top-3 Match (FS)"] = df_eval.apply(lambda row: top3_accuracy(row["Predicted Tags (Few-shot)"], row["Ticket Type"]), axis=1)

# Accuracy scores
print(f"\nFew-shot Top-1 Accuracy: {df_eval['Top-1 Match (FS)'].mean():.2%}")
print(f"Few-shot Top-3 Accuracy: {df_eval['Top-3 Match (FS)'].mean():.2%}")

# Save few-shot results
df_eval[[
    "Ticket Description",
    "Ticket Type",
    "Predicted Tags (Few-shot)",
    "Top-1 Match (FS)",
    "Top-3 Match (FS)"
]].to_csv("classified_tickets_few_shot_eval.csv", index=False)

print("\n🎉 Few-shot evaluation results saved to classified_tickets_few_shot_eval.csv")