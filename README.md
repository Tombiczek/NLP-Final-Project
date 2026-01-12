# Porównanie metod Parameter-Efficient Fine-tuning (PEFT) na wybranych zadaniach NLP

## O projekcie

Celem tego projektu jest zbadanie i porównanie efektywności różnych technik Parameter-Efficient Fine-Tuning (PEFT) w kontekście zadań przetwarzania języka naturalnego. W przeciwieństwie do pełnego dostrajania (Full Fine-Tuning), metody PEFT pozwalają na adaptację dużych modeli językowych przy znacznie mniejszym nakładzie obliczeniowym, modyfikując jedynie niewielki ułamek parametrów.

Projekt skupia się na dwóch odmiennych typach zadań NLP:
1.  **Klasyfikacja tekstu** – przy użyciu modelu **BERT** (zadanie: wykrywanie tekstu generowanego przez AI, zbiór AI-GA).
2.  **Streszczanie tekstu (Seq2Seq)** – przy użyciu modelu **mT5** (zadanie: wielojęzyczne streszczanie artykułów, zbiór XL-Sum w języku hiszpańskim).

W ramach eksperymentów porównywane są następujące metody:
*   **LoRA (Low-Rank Adaptation)**
*   **Adaptery** (architektury Houlsby oraz Pfeiffer)
*   **Prefix Tuning**
*   **Full Fine-Tuning** (jako punkt odniesienia – baseline)

Badania obejmują analizę jakości predykcji (metryki Accuracy, F1, ROUGE) oraz efektywności obliczeniowej (czas treningu, zużycie zasobów), a także weryfikację działania metod w warunkach ograniczonej liczby danych (*low-resource*).

## Wyniki 

Szczegółowe wyniki wszystkich eksperymentów, w tym wykresy funkcji straty i metryk w czasie rzeczywistym, dostępne są w serwisie [Weights & Biases](https://wandb.ai/tombik-warsaw-university-of-technology/nlp-project-peft/table?nw=nwusertombik).

## Struktura repozytorium

```bash
.
├── docs/               # Raport (pliki źródłowe LaTeX)
├── src/                # Kod źródłowy
│   ├── data.py         # Skrypty do ładowania i preprocessingu danych
│   ├── training.py     # Pętle treningowe i konfiguracja trenerów
│   └── utils.py        # Funkcje pomocnicze i narzędziowe
├── eksperymenty.ipynb  # Główny notebook Jupyter sterujący przebiegiem eksperymentów
└── README.md
```

## Uruchomienie

Cała logika eksperymentalna została zawarta w notatniku **`eksperymenty.ipynb`**. 

Aby uruchomić badania:
1.  Upewnij się, że posiadasz zainstalowane wymagane biblioteki (najbezpieczniej jest korzytstać z środowiska Google Colab).
2.  Otwórz plik `eksperymenty.ipynb` w środowisku Jupyter.
3.  Wykonuj kolejne komórki notebooka, które odpowiadają za:
    *   Wczytanie danych.
    *   Przygotowanie modeli.
    *   Trenowanie poszczególnych wariantów (Full FT, LoRA, Adaptery, Prefix).
    *   Ewaluację i prezentację wyników.

## Autorzy
*   Maja Zglinicka
*   Tomasz Lewiński
