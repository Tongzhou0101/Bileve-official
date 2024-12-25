import  torch

def calculate_ppl(cur_gen, model, tokenizer):
    tokd_all = tokenizer(cur_gen, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(tokd_all["input_ids"], labels=tokd_all["input_ids"])
        loss = outputs.loss
        ppl = torch.exp(loss)
    return ppl.item()

def nucleus_sampling(probs, p=0.9):
        # Sort the probabilities to identify top tokens
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Calculate cumulative probabilities and find cutoff threshold for top-p
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.where(cumulative_probs >= p)[1][0]

        # Consider only top-p tokens and re-normalize probabilities
        top_p_probs = sorted_probs[:, :cutoff_index + 1]
        top_p_probs /= torch.sum(top_p_probs, dim=-1, keepdim=True)
        return top_p_probs, sorted_indices
        