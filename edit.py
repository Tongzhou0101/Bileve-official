import torch
import random

import torch
import random

def substitution_attack(tokens, p, vocab_size, distribution=None):
    tokens = torch.tensor(tokens)  # Ensure tokens are a tensor
    if distribution is None:
        distribution = lambda x: torch.ones(size=(len(tokens), vocab_size)) / vocab_size
    idx = torch.randperm(len(tokens))[:int(p * len(tokens))]

    new_probs = distribution(tokens)
    samples = torch.multinomial(new_probs, 1).flatten()
    tokens[idx] = samples[idx]

    return tokens.tolist()  # Convert back to list if needed

def deletion_attack(tokens, p):
    tokens = torch.tensor(tokens)  # Ensure tokens are a tensor
    # print('ori',len(tokens))
    idx = torch.randperm(len(tokens))[:int(p * len(tokens))]
    # print('deletion',len(idx))
    
    keep = torch.ones(len(tokens), dtype=torch.bool)
    keep[idx] = False
    tokens = tokens[keep]
    
    return tokens.tolist()  # Convert back to list if needed

def insertion_attack(tokens, p, vocab_size, distribution=None):
    tokens = torch.tensor(tokens)  # Ensure tokens are a tensor
    if distribution is None:
        distribution = lambda x: torch.ones(size=(len(tokens), vocab_size)) / vocab_size
    idx = torch.randperm(len(tokens))[:int(p * len(tokens))]

    new_probs = distribution(tokens)
    samples = torch.multinomial(new_probs, 1).squeeze()

    for i in sorted(idx, reverse=True):  # Sort idx in descending order for correct insertion
        tokens = torch.cat([tokens[:i], samples[i:i+1], tokens[i:]])

    return tokens.tolist()  # Convert back to list if needed

def random_attack(tokens, p, vocab_size, distribution=None):
    # List of attack functions
    attacks = [
        lambda tokens: substitution_attack(tokens, p, vocab_size, distribution),
        lambda tokens: deletion_attack(tokens, p),
        lambda tokens: insertion_attack(tokens, p, vocab_size, distribution)
    ]
    
    # Randomly select an attack
    # attack = random.choice(attacks)
    probabilities = [0.8, 0.1, 0.1]
    
    # Randomly select an attack based on the defined probabilities
    attack = random.choices(attacks, probabilities)[0]
    
    
    # Execute the selected attack
    return attack(tokens)

