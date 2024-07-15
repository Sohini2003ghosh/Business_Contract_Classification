# Business_Contract_Classification
The first task is to parse these documents . Determine the key details within the contract document. Every contract has clauses  and sub-clauses. Use NER to determine the important feature from the document. Then Use Knowledge-Graph to see the important feature. Using Bert Model analyse the semantics and classify the contents of the parsed documents to these clauses.  A contract has an associated template to it, and it is important to determine the deviations  from that template and highlight them and analyse the semantics to generate corresponding suggestions.
# Code for the above problem
import random
import csv
import pdfkit
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import fitz
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification,DistilBertTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
from faker import Faker




# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Initialize Faker for synthetic contract generation
fake = Faker()

# Function to generate synthetic contract data
def generate_synthetic_contract():
    contract = {
        "Party A": fake.company(),
        "Party B": fake.company(),
        "Start Date": fake.date_between(start_date='-1y', end_date='+1y').strftime("%B %d, %Y"),
        "End Date": fake.date_between(start_date='+1y', end_date='+5y').strftime("%B %d, %Y"),
        "Amount": f"${random.randint(1000, 100000)}",
        "Notice Period (days)": random.choice([30, 60, 90]),
        "Confidentiality Clause": fake.sentence(nb_words=10),
        "Payment Terms": f"Party B agrees to pay Party A a sum of ${random.randint(1000, 100000)} for the services provided. Payment shall be made within {random.choice([30, 60, 90])} days.",
        "Termination Clause": f"Either party may terminate this Contract with {random.choice([30, 60, 90])} days written notice.",
        "Liability Clause": "Party A shall not be liable for any damages exceeding the amount paid by Party B."
    }
    return contract

# Generate synthetic contracts and save to CSV
num_contracts = 100
contracts = [generate_synthetic_contract() for _ in range(num_contracts)]
csv_file_path = 'business_advanced_contracts_dataset.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=contracts[0].keys())
    writer.writeheader()
    for contract in contracts:
        writer.writerow(contract)
print(f"CSV file generated at: {csv_file_path}")

# Function to create PDF from contract
def create_pdf(contract, pdf_file_path):
    c = canvas.Canvas(pdf_file_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, height - 30, "Business Contract")

    c.setFont("Helvetica", 10)
    y = height - 50
    for key, value in contract.items():
        c.drawString(30, y, f"{key}: {value}")
        y -= 20

    c.save()

# Generate PDFs from the contracts
for i, contract in enumerate(contracts):
    pdf_file_path = f'contract_advanced_feature{i + 1}.pdf'
    create_pdf(contract, pdf_file_path)
    print(f"PDF file generated at: {pdf_file_path}")
    





# Sample clauses for training
clauses = ["Party A", "Party B", "Start Date", "End Date", "Amount", "Notice Period (days)",
           "Confidentiality Clause", "Payment Terms", "Termination Clause", "Liability Clause"]


# Load the dataset
df = pd.read_csv(csv_file_path)

# Preprocess the dataset
texts = []
labels = []
label_map = {clause: idx for idx, clause in enumerate(df.columns)}

for _, row in df.iterrows():
    for clause, text in row.items():
        if clause in label_map:
            texts.append(f"{clause}: {text}")
            labels.append(label_map[clause])

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class ContractDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = ContractDataset(texts, labels)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Define suggestions for each type of deviation
suggestions = {
    "Invalid date format": "Please provide the date in the format 'Month Day, Year' (e.g., January 1, 2024).",
    "Dates order incorrect": "The Start Date should be before the End Date. Please adjust accordingly.",
    "Invalid company name": "Ensure the company name contains only letters, numbers, spaces, and certain punctuation.",
    "Invalid amount format": "The amount should start with a '$' sign and only contain digits and commas (if applicable).",
    "Negative amount": "The amount cannot be negative. Please enter a valid non-negative amount.",
    "Invalid notice period": "The notice period should be a positive integer indicating days.",
    "Incorrect clause category": "It seems the clause is categorized incorrectly. Please check and categorize properly."
}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to perform NER on text
def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to extract clauses from text
def extract_clauses_from_text(text):
    lines = text.split('\n')
    extracted_clauses = {}
    for line in lines:
        for clause in template.keys():
            if line.startswith(clause):
                extracted_clauses[clause] = line.split(":", 1)[1].strip()
                break
    return extracted_clauses

# Function to generate knowledge graph
def generate_knowledge_graph(entities):
    G = nx.Graph()
    for entity, label in entities:
        G.add_node(entity, label=label)
    for i in range(len(entities) - 1):
        G.add_edge(entities[i][0], entities[i + 1][0])
    return G

# Function to visualize knowledge graph
def visualize_knowledge_graph(G):
    pos = nx.spring_layout(G, k=1, iterations=100)  # Adjust k and iterations for better spacing

    plt.figure(figsize=(15, 10))

    node_labels = nx.get_node_attributes(G, 'label')
    node_colors = [plt.cm.Paired(i) for i in range(len(node_labels))]
    edge_colors = ['gray' for _ in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.9, linewidths=1, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif', font_weight='bold')

    for node, (x, y) in pos.items():
        plt.text(x, y, node_labels[node], fontsize=12, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.6))

    plt.title('Knowledge Graph from Contract', fontsize=15)
    plt.axis('off')  # Hide axis
    plt.show()

# Helper functions for deviation detection
def is_valid_company_name(name):
    return name.replace(" ", "").replace(".", "").isalnum()

def is_valid_date(date_str):
    try:
        datetime.strptime(date_str, "%B %d, %Y")
        return True
    except ValueError:
        return False

# Function to detect deviations and generate suggestions with semantic understanding
def detect_deviation_with_semantics(template, contract):
    deviations = []

    def add_deviation(deviation_type, clause, contract_text):
        deviations.append({
            "type": deviation_type,
            "clause": clause,
            "contract_text": contract_text,
            "suggestion": suggestions.get(deviation_type, "No suggestion available.")
        })

    # Check if all required clauses are present
    for clause, template_text in template.items():
        if clause not in contract:
            add_deviation("Missing clause", clause, "Clause not found in contract.")
        else:
            contract_text = contract[clause]

#             Check for company names validity
            if clause in ["Party A", "Party B"]:
                if not is_valid_company_name(contract_text):
                    add_deviation("Invalid company name", clause, contract_text)

            # Check for valid dates and their order
            if clause in ["Start Date", "End Date"]:
                
                if not is_valid_date(contract_text):
                    add_deviation("Invalid date format", clause, contract_text)
                elif clause == "Start Date" and "End Date" in contract:
                    start_date = datetime.strptime(contract_text, "%B %d, %Y")
                    end_date = datetime.strptime(contract["End Date"], "%B %d, %Y")
                    if start_date >= end_date:
                        add_deviation("Dates order incorrect", clause, contract_text)

            # Check for valid amount (non-negative)
            elif clause == "Amount":
                
                if not contract_text.startswith("$") or not contract_text[1:].replace(",", "").isdigit():
                    add_deviation("Invalid amount format", clause, contract_text)
                elif int(contract_text[1:].replace(",", "")) < 0:
                    add_deviation("Negative amount", clause, contract_text)

            # Check for notice period to be a valid number
            elif clause == "Notice Period (days)":
                if not contract_text.isdigit() or int(contract_text) < 0:
                    add_deviation("Invalid notice period", clause, contract_text)

            # Check for incorrect clause category
            if clause in ["Confidentiality Clause", "Payment Terms", "Termination Clause", "Liability Clause"]:
                if "Clause" in clause and "Clause" not in contract_text:
                    add_deviation("Incorrect clause category", clause, contract_text)

    return deviations

# Function to highlight deviations and save to PDF
def highlight_deviation(template, contract):
    highlighted_html = "<html><body><h1>Contract with Highlighted Deviations</h1><pre>"

    for clause, template_text in template.items():
        if clause not in contract:
            highlighted_html += f"<span style='background-color: yellow;'>Missing clause: {clause}</span>\n"
        else:
            contract_text = contract[clause]
            if template_text != contract_text:
                highlighted_html += f"<span style='background-color: yellow;'>Deviation in clause {clause}: {contract_text}</span>\n"
            else:
                highlighted_html += f"{clause}: {contract_text}\n"

    highlighted_html += "</pre></body></html>"
    return highlighted_html

# Sample template
template = {
    "Party A": "Alpha Corp",
    "Party B": "Beta LLC",
    "Start Date": "March 15, 2024",
    "End Date": "March 14, 2025",
    "Amount": "$20,000",
    "Notice Period (days)": 30,
    "Confidentiality Clause": "Party A and Party B agree to maintain the confidentiality of all information shared.",
    "Payment Terms": "Party B agrees to pay Party A a sum of $20,000 for the services provided. Payment shall be made.",
    "Termination Clause": "Either party may terminate this Contract with 30 days written notice.",
    "Liability Clause": "Party A shall not be liable for any damages exceeding the amount paid by Party B."
}

# Function to analyze PDF and test contract
def analyze_pdf_and_test_contract(pdf_path, template):
    # Extract text from PDF
    contract_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted text from {pdf_path}:\n{contract_text}\n")

    # Perform NER on extracted text
    entities = perform_ner(contract_text)
    print(f"Entities: {entities}\n")

    # Generate knowledge graph
    G = generate_knowledge_graph(entities)
    visualize_knowledge_graph(G)

    # Extract clauses from text
    test_contract = extract_clauses_from_text(contract_text)

    # Detect deviations with semantics and suggestions
    deviations = detect_deviation_with_semantics(template, test_contract)
    for deviation in deviations:
        print(f"{deviation['type']} in clause {deviation['clause']}: {deviation['contract_text']}")
        print(f"Suggestion: {deviation['suggestion']}\n")

    # Highlight deviations and save to PDF
    highlighted_html = highlight_deviation(template, test_contract)
    highlighted_pdf_path = f"highlighted_{pdf_path.replace('.pdf', '')}.pdf"
    with open("highlighted_contract.html", "w") as file:
        file.write(highlighted_html)

    pdfkit.from_file("highlighted_contract.html", highlighted_pdf_path)
    print(f"Highlighted PDF file generated at: {highlighted_pdf_path}")

# Take user input for PDF file path
pdf_path = input("Enter the path of the PDF contract file: ")

# Analyze the user-provided PDF and test contract
analyze_pdf_and_test_contract(pdf_path, template)
