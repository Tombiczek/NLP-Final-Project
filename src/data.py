import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, load_dataset

AI_GA_URL = (
    "https://raw.githubusercontent.com/panagiotisanagnostou/AI-GA/"
    "refs/heads/main/ai-ga-dataset.csv"
)

def load_ai_ga(csv_url = AI_GA_URL, seed = 42,) -> DatasetDict:
    print("Downloading and processing dataset...")

    df = pd.read_csv(csv_url)

    # Upewniamy się, że kolumna label jest typu int
    df["label"] = df["label"].astype(int)

    # 3. Podział na oryginały i abstrakty generowane przez AI
    original = df[df["label"] == 0]
    ai = df[df["label"] == 1]

    # Docelowy podział: 80% train, 10% val, 10% test
    train_frac = 0.8
    val_frac = 0.1
    test_frac = 0.1

    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Podziały muszą sumować się do 1.0")

    # 4. Logika podziału
    ai_train = ai.sample(frac=train_frac, random_state=seed)
    remaining_ai = ai[~ai["title"].isin(ai_train["title"])]

    val_relative_frac = val_frac / (val_frac + test_frac)
    ai_val = remaining_ai.sample(frac=val_relative_frac, random_state=seed)
    ai_test = remaining_ai[~remaining_ai["title"].isin(ai_val["title"])]

    original_train = original[original["title"].isin(ai_train["title"])]
    original_val = original[original["title"].isin(ai_val["title"])]
    original_test = original[original["title"].isin(ai_test["title"])]

    train_df = pd.concat([original_train, ai_train]).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat([original_val, ai_val]).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat([original_test, ai_test]).sample(frac=1, random_state=seed).reset_index(drop=True)

    # 5. Definicja Features (Ważne dla HuggingFace Trainer)
    features = Features({
        "title": Value("string"),
        "abstract": Value("string"),
        "label": ClassLabel(num_classes=2, names=["original", "ai_generated"])
    })

    cols = ["title", "abstract", "label"]

    train_ds = Dataset.from_pandas(train_df[cols], features=features, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[cols], features=features, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df[cols], features=features, preserve_index=False)

    dataset_dict = DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
    )
    return dataset_dict

def load_xlsum(language) -> DatasetDict:

    dataset = load_dataset("csebuetnlp/xlsum", language)

    return dataset

