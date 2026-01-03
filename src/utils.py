from datasets import DatasetDict
import numpy as np

def average_length_ai_ga(ds_ai_ga):
    train = ds_ai_ga["train"]
    lengths = [len(x.split()) for x in train["abstract"]]
    avg_len = np.mean(lengths)
    print(f"Średnia długość abstraktu w AI-GA (w słowach): {avg_len:.2f}")

# ------------------------------------------------------------
# 2. ŚREDNIA DŁUGOŚĆ ARTYKUŁÓW I STRESZCZEŃ W XL-SUM
# ------------------------------------------------------------

def average_length_xlsum(ds, name=""):
    train = ds["train"]
    text_lengths = [len(t.split()) for t in train["text"]]
    summary_lengths = [len(s.split()) for s in train["summary"]]

    print(f"\n{name}:")
    print(f"  Średnia długość artykułu: {np.mean(text_lengths):.2f} słów")
    print(f"  Średnia długość streszczenia: {np.mean(summary_lengths):.2f} słów")

# ------------------------------------------------------------
# 3. WYŚWIETLENIE JEDNEGO PRZYKŁADU Z KAŻDEGO DATASETU
# ------------------------------------------------------------

def show_sample(ds, name, idx=0):
    print(f"\n=== SAMPLE FROM {name.upper()} ===")
    example = ds["train"][idx]
    for k, v in example.items():
        if isinstance(v, str):
            print(f"{k}: {v[:300]}{'...' if len(v) > 300 else ''}")
        else:
            print(f"{k}: {v}")
