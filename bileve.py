import torch
from fastecdsa import ecdsa
from utils import nucleus_sampling




def generate(logger, private_key, tokenizer, model, prompt, vocab_size, n, m, xi, d, gamma, mode):
    # Prepare the input prompt and other initial variables
    inputs = prompt.to(model.device)

    shift = torch.randint(n, (1,))  # Random shift for generating tokens to improve diversity
    attn = torch.ones_like(inputs) 
    past = None  
    msg_tokens = []  # List to store message tokens

    with torch.no_grad():
        for i in range(m + d):
            
            output = model(inputs[:, -1:], past_key_values=past, attention_mask=attn) if past else model(inputs)
            probs = torch.nn.functional.softmax(output.logits[:, -1, :vocab_size], dim=-1).cpu()  

            if i < d:
                # Tokens generated during the first 'd' iterations are detmermined as the message
                token = exp_sampling(probs, xi[(shift + i) % n, :]).to(model.device)
                msg_tokens.append(token)
            else:
                if i == d:
                    # Generate the signature from the message tokens
                    msg_str = tokenizer.decode(torch.cat(msg_tokens, dim=-1)[0])

                    # Sign the message string using the private key. It can be replaced with any other signing method.
                    r, s = ecdsa.sign(msg_str, private_key)  
                    signature = [
                        int(char) for char in format(r, '0256b')
                    ] + [
                        int(char) for char in format(s, '0256b')
                    ]

                    logger.info('r_g: %s', r)
                    logger.info('s_g: %s', s)

                # Generate watermarked tokens with the signature embedded
                token = my_sampling(probs, xi[(shift + i) % n, :], signature[i - d], vocab_size, gamma, mode).to(model.device)

            inputs = torch.cat([inputs, token], dim=-1)  

            past = output.past_key_values  
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1) 

    return inputs[:, -m-d:], signature  # Return the watermarked sequence and the signature

def exp_sampling(probs, u):
    # Exponential sampling based on the probability distribution
    return torch.argmax(u ** (1 / probs), axis=1).unsqueeze(-1)

def my_sampling(probs, u, b, vocab_size, gamma, mode):
    if mode == 'SLS': # single-level signature
        top_p_probs, sorted_indices = nucleus_sampling(probs)

        while True:
            # Sample tokens from the top-p probabilities
            next_tokens = torch.multinomial(top_p_probs[:,], 20, replacement=False)
            next_token_ids = sorted_indices[:, next_tokens]

            # Here we use simple mapping (%2) to replace hash function for simplicity. 
            # Any hash function splitting the token space into two parts can be used. 
            # The security of the hash function would increase the overall security of the watermarking scheme.
            if (next_token_ids % 2 == b).any():
                t = next_token_ids[next_token_ids % 2 == b][0]
                return torch.tensor([[t]])
            
    if mode == 'bileve':
        # Rank-based sampling used in the bi-level signature
        rank_probs = torch.argsort(torch.argsort(probs)) / vocab_size  # Higher probabilities have higher rank scores
        rank_u = torch.argsort(torch.argsort(u)) / vocab_size  # Rank scores for key sequence

        # Embed coarse-grained signature 
        combined_ranks = rank_probs + gamma * rank_u  
        sorted_indices = torch.topk(combined_ranks, 50, dim=1).indices

        for t in sorted_indices[0]:
            # Embed fine-grained signature, b is the signature bit for the current token
            if t % 2 == b:  # Replace hash with a simple mapping for simplicity
                return torch.tensor([[t]])







