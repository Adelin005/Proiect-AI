import pandas as pd
import numpy as np
import random
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Setarea seed-ului pentru reproductibilitate
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Detectează dispozitivul (CPU sau GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispozitiv folosit pentru antrenare: {device}")


# --- 1. Colectarea și Preprocesarea Datelor (Etapa 1) ---

FILE_PATH = "Tweets.csv"
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(
        f"Eroare: Fișierul '{FILE_PATH}' nu a fost găsit. Asigură-te că este în director."
    )
    exit()

# Selecția coloanelor și curățarea inițială
df = df[["airline_sentiment", "text"]]


# Curățarea datelor: Elimină mențiunile (@user) și link-urile
def clean_text(text):
    text = re.sub(r"http\S+|@\w+", "", text)
    # Conversia la litere mici și eliminarea spațiilor multiple
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["text"] = df["text"].apply(clean_text)

# Maparea etichetelor la numere (obligatoriu pentru PyTorch)
# NEGATIVE (0), NEUTRAL (1), POSITIVE (2)
sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
df["labels"] = df["airline_sentiment"].map(sentiment_map)

# Păstrăm și distribuția originală pentru vizualizare în Streamlit
sentiment_distribution_df = (
    df["airline_sentiment"].value_counts(normalize=True).reset_index()
)
sentiment_distribution_df.columns = ["Sentiment", "Procent"]
sentiment_distribution_df["Procent"] = (
    sentiment_distribution_df["Procent"] * 100
).round(2)
# Salvăm distribuția într-un fișier CSV pentru a fi încărcată în app.py
sentiment_distribution_df.to_csv("sentiment_distribution.csv", index=False)
print("Distribuția sentimentelor a fost salvată în 'sentiment_distribution.csv'.")


# Împărțirea datelor: 80% antrenare, 20% testare
texts = df["text"].tolist()
labels = df["labels"].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=SEED, stratify=labels
)


# --- 2. Reprezentarea Numerică (Embeddings și Tokenizare) ---

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenizarea seturilor de date
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)


# Crearea clasei PyTorch Dataset
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)


# --- 3. Antrenarea Modelului AI (Finetuning) ---

# Încărcarea modelului pre-antrenat pentru clasificare secvențială (3 clase)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
).to(device)


# Definirea funcției de evaluare (necesară pentru Trainer)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Suportul include 3 clase: 0=Negative, 1=Neutral, 2=Positive
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Definirea hiperparametrilor de antrenare
training_args = TrainingArguments(
    output_dir="./results",  # Director de salvare a rezultatelor
    num_train_epochs=3,  # Număr de epoci (suficient pentru finetuning)
    per_device_train_batch_size=16,  # Mărimea batch-ului de antrenare
    per_device_eval_batch_size=64,  # Mărimea batch-ului de evaluare
    warmup_steps=500,  # Etape de încălzire a ratei de învățare
    weight_decay=0.01,  # Regularizare L2
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",  # <--- CORECȚIE AICI: 'evaluation_strategy' devine 'eval_strategy'
    save_strategy="epoch",  # Salvează modelul la sfârșitul fiecărei epoci
    load_best_model_at_end=True,  # Încarcă cel mai bun model bazat pe metrică
    report_to="none",  # Nu raportează la Weights & Biases sau alte platforme
)
# Inițializarea Trainer-ului
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Rularea antrenării
print("\n--- Începe Antrenarea Modelului DistilBERT (3 epoci) ---")
trainer.train()

# --- 4. Evaluarea și Testarea Finală ---

print("\n--- Evaluare Finală pe Setul de Testare ---")
results = trainer.evaluate()

print("=" * 40)
print("     REZULTATE PERFORMANȚĂ MODEL FIN-TUNAT     ")
print("=" * 40)
print(f"Acuratețe (Accuracy): {results['eval_accuracy']:.4f}")
print(f"Scor F1 (F1-Score): {results['eval_f1']:.4f}")
print(f"Precision: {results['eval_precision']:.4f}")
print(f"Recall: {results['eval_recall']:.4f}")
print(f"Loss mediu: {results['eval_loss']:.4f}")
print("=" * 40)


# Salvarea modelului fin-tunat (pentru a-l folosi în interfața Streamlit)
model_path = "./sentiment_model_finetuned"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
print(f"\nModelul și Tokenizer-ul au fost salvate în: {model_path}")
