# NLP Final Project

## Wstępna struktura
```bash
nlp-peft-project/
│
├── src/
│   ├── data.py                # Wczytanie + tokenizacja AI-GA i XL-Sum
│   ├── models.py              # BERT, mT5 + PEFT (LoRA/Prefix/Adaptery)
│   ├── training.py            # prosty trainer dla klasyfikacji i seq2seq
│   └── utils.py               # helpery: seed, device, proste metryki
│
└── wstepne_wyniki.ipynb
```