!!! IMPORTANT - "Folderul sentiment_model_finetuned nu este complet pe GitHub din cauza limitărilor de mărime. Modelul trebuie generat local prin rularea scriptului train_model.py." !!!
RULARE - python -m streamlit run app.py

# ✈️ Airline Sentiment Analysis Dashboard

Acest proiect reprezintă o soluție avansată de Inteligență Artificială destinată clasificării automate a sentimentelor din postările de pe rețelele sociale (Twitter). Utilizând arhitectura **DistilBERT** și tehnici de **Fine-Tuning**, sistemul analizează feedback-ul clienților companiilor aeriene, categorisindu-l în: **Negativ**, **Neutru** sau **Pozitiv**.



## 📋 Structura Proiectului

* `app.py`: Aplicația web principală dezvoltată în **Streamlit**. Gestionează interfața utilizator și inferența modelului.
* `train_model.py`: Scriptul de antrenare care procesează datele, efectuează fine-tuning-ul modelului și salvează rezultatele.
* `Tweets.csv`: Setul de date (dataset) original conținând mii de tweet-uri etichetate.
* `sentiment_distribution.csv`: Fișier generat automat ce conține statistici despre distribuția claselor din dataset.
* `sentiment_model_finetuned/`: Folderul care găzduiește modelul antrenat și tokenizer-ul salvat.

## 🚀 Instalare și Configurare

### 1. Descărcarea Proiectului
Asigură-te că toate fișierele menționate mai sus se află în același director de lucru.

### 2. Instalarea Dependențelor
Proiectul necesită Python 3.8+. Instalează librăriile necesare rulând următoarea comandă în terminal:

pip install streamlit torch transformers pandas numpy altair scikit-learn accelerate

🧠 Mod de Utilizare
Pasul 1: Antrenarea Modelului (Opțional)
Dacă nu ai deja folderul sentiment_model_finetuned, rulează procesul de antrenare:
python train_model.py

Pasul 2: Lansarea Dashboard-ului
Pornește interfața grafică interactivă cu următoarea comandă:

streamlit run app.py
