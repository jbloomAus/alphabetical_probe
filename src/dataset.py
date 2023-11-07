import re
import nltk
import torch 
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    WeightedRandomSampler,
    random_split,

)

def get_classification_indices(vocab, regex_pattern):
    # get indices of words that match the regex pattern
    indices = [
        v for k, v in vocab.items() if re.match(regex_pattern, k.strip().strip("Ġ"))
    ]
    # get the indices of the words that don't match the regex pattern
    not_indices = [
        v for k, v in vocab.items() if not re.match(regex_pattern, k.strip().strip("Ġ"))
    ]
    return indices, not_indices


def get_regex_pattern(letter, criterion):
    # Escaping the letter in case it has a special meaning in regex (e.g., '.' or '*')
    escaped_letter = re.escape(letter)

    if criterion == "contains_any":
        # Match the letter anywhere in the string
        return rf"{escaped_letter}"
    elif criterion == "starts":
        # Match strings that start with the letter
        return rf"^{escaped_letter}"
    elif criterion == "ends":
        # Match strings that end with the letter
        return rf"{escaped_letter}$"
    elif criterion == "contains_1":
        # Match strings that contain the letter exactly once
        return rf"^[^{escaped_letter}]*{escaped_letter}[^{escaped_letter}]*$"
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def make_weights_for_balanced_classes(labels):
    # Count of instances in each class
    n_total = len(labels)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos


    # Weight for each sample
    weight_per_class = {True: n_total / n_pos, False: n_total / n_neg}

    # Assign weight to each sample
    weights = [weight_per_class[label.item()] for label in labels]
    
    return weights

def filter_vocab(vocab):
    roman_char_regex = re.compile('^[A-Za-z]+$')
    words = nltk.corpus.words
    english_words = set(w.lower() for w in words.words())
    new_vocab = {}
    for word, index in vocab.items():
        clean_word = word.lstrip("Ġ").strip().lower()
        if clean_word in english_words and bool(roman_char_regex.match(clean_word)):
            new_vocab[word] = index
    return new_vocab


def get_letter_dataset(
    criterion,
    target,
    embeddings,
    vocab,
    batch_size=32,
    rebalance=False,
    test_proportion=0.2,
):
    
    
    new_vocab = filter_vocab(vocab)
    vocab_tokens = list(new_vocab.keys())
    new_embeddings = embeddings[list(new_vocab.values())]
    # original_indices = list(new_vocab.values())
    # index_mapping = {v: i for i, (k, v) in enumerate(new_vocab.items())}

    # get indices of words that start with the letter A
    regex_pattern = get_regex_pattern(target, criterion)
    labels = [bool(re.search(regex_pattern, k.strip().strip("Ġ"))) for k in vocab_tokens]
    labels = torch.tensor(labels, dtype=torch.bool, device=new_embeddings.device, requires_grad=False)
    
    dataset = TensorDataset(new_embeddings, labels)
    
    train_size = int((1 - test_proportion) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # rebalance
    if rebalance:
        train_labels = dataset.tensors[1][train_dataset.indices]
        train_weights = make_weights_for_balanced_classes(train_labels)
        sampler = WeightedRandomSampler(train_weights, len(train_weights))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    return train_loader, test_loader

