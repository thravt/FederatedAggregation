import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##########################################################################
# Utility: Create a Balanced Subset
##########################################################################
def get_balanced_subset(dataset, desired_total):
    """
    From the given dataset, select a balanced subset.
    If there aren't enough samples to achieve `desired_total`, then use the maximum
    number possible (2 * min{#negatives, #positives}).
    """
    labels = dataset["label"]
    pos_idx = [i for i, v in enumerate(labels) if v == 1]
    neg_idx = [i for i, v in enumerate(labels) if v == 0]
    max_possible = 2 * min(len(pos_idx), len(neg_idx))
    if desired_total > max_possible:
        print(f"Not enough samples to create a balanced subset of size {desired_total}. " \
              f"Using maximum balanced size: {max_possible}")
        desired_total = max_possible
    num_each = desired_total // 2
    selected_pos = np.random.choice(pos_idx, num_each, replace=False).tolist()
    selected_neg = np.random.choice(neg_idx, num_each, replace=False).tolist()
    selected_indices = selected_pos + selected_neg
    return dataset.select(selected_indices)

##########################################################################
# 1. Load, Preprocess, and Balance the Dataset (Sentiment140 Subset)
##########################################################################
# Load the Sentiment140 dataset.
dataset = load_dataset("sentiment140")
# Shuffle and restrict to 2,000 samples for faster training, ensuring a mix of classes.
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(2000))
# Debug: show raw distribution of original sentiment labels (0=neg, 4=pos).
print("Raw 2000-sample sentiment distribution:", Counter(dataset["train"]["sentiment"]))

# Preprocessing function: try multiple potential keys.
def preprocess(example):
    if "target" in example:
        original = example["target"]
    elif "polarity" in example:
        original = example["polarity"]
    elif "sentiment" in example:
        original = example["sentiment"]
    elif "label" in example:
        original = example["label"]
    else:
        raise KeyError("No valid sentiment key found in the example.")
    # Map 0 → 0 (negative) and any non-zero value → 1 (positive)
    example["label"] = 0 if original == 0 else 1
    return example

# Apply preprocessing to map labels to {0,1}.
dataset = dataset.map(preprocess)
# Debug: show distribution after mapping labels.
print("Post-preprocess label distribution:", Counter(dataset["train"]["label"]))

# Create a balanced subset. If 2000 balanced examples are unavailable, use the maximum possible.
balanced_dataset = get_balanced_subset(dataset["train"], 2000)
print("Balanced subset size:", len(balanced_dataset))

# Split the balanced dataset into client train (60%) and temp (40%).
split1 = balanced_dataset.train_test_split(test_size=0.4, seed=42)
client_train_dataset = split1["train"]
temp_dataset = split1["test"]
# Further split temp into unlabeled (50%) and test (50%).
temp_split = temp_dataset.train_test_split(test_size=0.5, seed=42)
unlabeled_dataset = temp_split["train"]
test_dataset = temp_split["test"]

print("Client train distribution:", Counter(client_train_dataset["label"]))
print("Unlabeled distribution:", Counter(unlabeled_dataset["label"]))
print("Test distribution:", Counter(test_dataset["label"]))

##########################################################################
# 2. Tokenize the Data with a Lightweight Transformer
##########################################################################
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# Tokenize and clean each split
cols_to_keep = ["input_ids", "attention_mask", "label"]
for split_name in ["client_train", "unlabeled", "test"]:
    ds = eval(f"{split_name}_dataset")
    ds = ds.map(tokenize_function, batched=True)
    drop = [c for c in ds.column_names if c not in cols_to_keep]
    ds = ds.remove_columns(drop)
    ds.set_format(type="torch", columns=cols_to_keep)
    globals()[f"{split_name}_dataset"] = ds

# Create DataLoaders
batch_size = 16
client_loaders = []
# Partition client_train into smaller client datasets
num_clients = 3
def partition_dataset(dataset, num_clients, portion_size=100):
    size = len(dataset)
    idxs = np.random.permutation(size)
    parts = np.array_split(idxs, num_clients)
    clients = []
    for part in parts:
        sub = dataset.select(part.tolist())
        try:
            sub = get_balanced_subset(sub, min(portion_size, len(sub)))
        except ValueError:
            pass
        clients.append(sub)
    return clients
client_datasets = partition_dataset(client_train_dataset, num_clients, portion_size=100)
for ds in client_datasets:
    client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
test_loader = DataLoader(test_dataset, batch_size=batch_size)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

##########################################################################
# 3. Partition the Client Training Data Among Clients (Small, Disjoint Portions)
##########################################################################
num_clients = 3

def partition_dataset(dataset, num_clients, portion_size=100):
    size = len(dataset)
    indices = np.arange(size)
    np.random.shuffle(indices)
    client_indices = np.array_split(indices, num_clients)

    client_datasets = []
    for idx in client_indices:
        client_data = dataset.select(idx.tolist())
        try:
            balanced_client_data = get_balanced_subset(client_data, min(portion_size, len(client_data)))
        except ValueError:
            balanced_client_data = client_data
        client_datasets.append(balanced_client_data)
    return client_datasets

client_datasets = partition_dataset(client_train_dataset, num_clients, portion_size=100)

batch_size = 16

def create_dataloader(ds):
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

client_loaders = [create_dataloader(ds) for ds in client_datasets]
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

##########################################################################
# 4. Define Helper Functions for Local Training & Evaluation
##########################################################################
def local_train(model, dataloader, epochs=1, lr=2e-5):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    return model.state_dict()


def evaluate(model, dataloader):
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item() * input_ids.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="weighted")
    return avg_loss, acc, f1

##########################################################################
# 5. Federated Aggregation Methods
##########################################################################
def fedavg_aggregate(global_model, client_state_dicts, client_sizes):
    new_state = copy.deepcopy(global_model.state_dict())
    total = sum(client_sizes)
    for key in new_state.keys():
        aggregated = sum((size/total) * state[key] for state, size in zip(client_state_dicts, client_sizes))
        new_state[key] = aggregated
    return new_state


def fedlama_aggregate(global_model, client_state_dicts, client_sizes):
    new_state = copy.deepcopy(global_model.state_dict())
    total = sum(client_sizes)
    def get_alpha(k):
        if "classifier" in k: return 0.8
        if "embeddings" in k: return 0.2
        return 0.5
    for k in new_state.keys():
        alpha = get_alpha(k)
        weighted = sum((size/total) * state[k] for state, size in zip(client_state_dicts, client_sizes))
        new_state[k] = (1-alpha)*new_state[k] + alpha*weighted
    return new_state


def feddist_aggregate(global_model, client_models, unlabeled_loader, distill_steps=1, lr=2e-5):
    global_model.train()
    optimizer = optim.Adam(global_model.parameters(), lr=lr)
    temperature = 2.0
    kd_loss_fn = nn.KLDivLoss(reduction="batchmean")
    for _ in range(distill_steps):
        for batch in unlabeled_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                soft_sum = 0
                for m in client_models:
                    m.eval()
                    out = m(input_ids=input_ids, attention_mask=attention_mask)
                    soft = torch.log_softmax(out.logits/temperature, dim=-1)
                    soft_sum += soft
                avg_soft = soft_sum / len(client_models)
            optimizer.zero_grad()
            out_g = global_model(input_ids=input_ids, attention_mask=attention_mask)
            student = torch.log_softmax(out_g.logits/temperature, dim=-1)
            loss = kd_loss_fn(student, avg_soft) * (temperature**2)
            loss.backward()
            optimizer.step()
    return global_model.state_dict()

##########################################################################
# 6. Federated Training Loop
##########################################################################
def federated_training(aggregation_method, rounds=3):
    config = AutoConfig.from_pretrained(model_checkpoint, num_labels=2)
    global_model = AutoModelForSequenceClassification.from_config(config).to(device)
    global_state = global_model.state_dict()

    history = []
    for rnd in range(1, rounds+1):
        print(f"\nFederated Round {rnd}/{rounds}")
        states, sizes, models = [], [], []
        for loader in client_loaders:
            m = AutoModelForSequenceClassification.from_config(config).to(device)
            m.load_state_dict(global_state)
            states.append(local_train(m, loader, epochs=1, lr=2e-5))
            sizes.append(len(loader.dataset))
            models.append(m)

        if aggregation_method == "fedavg":
            global_state = fedavg_aggregate(global_model, states, sizes)
        elif aggregation_method == "fedlama":
            global_state = fedlama_aggregate(global_model, states, sizes)
        elif aggregation_method == "feddist":
            global_state = feddist_aggregate(global_model, models, unlabeled_loader, distill_steps=1, lr=2e-5)
        else:
            raise ValueError("Unknown aggregation method.")
        global_model.load_state_dict(global_state)
        loss, acc, f1 = evaluate(global_model, test_loader)
        history.append({"round": rnd, "loss": loss, "accuracy": acc, "f1": f1})
        print(f"Test Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}")
    return history

##########################################################################
# 7. Run Experiments for Each Aggregation Method and Report Metrics
##########################################################################
if __name__ == "__main__":
    methods = ["fedavg", "fedlama", "feddist"]
    results = {}
    for method in methods:
        print(f"\n--- Training with {method.upper()} ---")
        metrics = federated_training(method, rounds=10)
        results[method] = metrics

    print("\n=== Federated Learning Experiment Results ===")
    for method, metrics in results.items():
        print(f"\nMethod: {method.upper()}")
        for m in metrics:
            print(m)
