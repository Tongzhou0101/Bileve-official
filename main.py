import os, argparse
import logging 
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from mersenne import mersenne_rng
from fastecdsa import curve, ecdsa, keys
from utils import calculate_ppl

from bileve import generate
from detect import permutation_test, verify


logger = logging.getLogger(__name__)
log_format = '[%(asctime)s] - %(message)s'
date_format='%Y/%m/%d %H:%M:%S'
formatter = logging.Formatter(log_format, date_format)
logger.setLevel(logging.INFO)




def main(args):
    # Create output directory if it does not exist
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, args.logfile)  # Set up the logfile path

    # Configure logging to write to file
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(args)  # Log the arguments

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')  # Select the device

    # ====================== Generation ======================
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    # Encode the prompt into tokens
    tokens = tokenizer.encode(args.prompt, return_tensors='pt', add_special_tokens=True)

    # Generate private and public keys using elliptic curve cryptography
    # One can replace this with any other signing method
    private_key = keys.gen_private_key(curve.P256)
    public_key = keys.get_public_key(private_key, curve.P256)

    # Initialize RNG with the provided key
    rng = mersenne_rng(args.key)
    vocab_size = len(tokenizer)  # Determine the vocabulary size

    # Generate a random tensor xi used for watermarking
    xi = torch.tensor([rng.rand() for _ in range(args.n * vocab_size)]).view(args.n, vocab_size)

    # Generate watermarked tokens and the signature
    watermarked_tokens, sig_g = generate(
        logger, private_key, tokenizer, model, tokens, vocab_size, args.n, args.m, xi, args.d, args.gamma, args.mode
    )

    # Decode watermarked tokens into text
    watermarked_text = tokenizer.decode(watermarked_tokens[0])
    logger.info('watermark_text: %s', watermarked_text)  # Log the watermarked text

    with open(os.path.join(args.out_dir, 'doc.txt'), 'w') as f:
        f.write(watermarked_text)

    print(watermarked_text)  # Print the watermarked text


    # ====================== Detection ======================

    # Verify the watermarked tokens using the public key
    valid = verify(logger, watermarked_tokens[0], public_key, tokenizer, args.m, args.d)
    if valid:
        logger.info('The verification is successful')  
    else:
        logger.info('Failure')  
        
        # If failed, perform permutation test
        with open(os.path.join(args.out_dir, 'doc.txt'), 'r') as f:
                text = f.read()
        tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]
        pval = permutation_test(tokens,args.key,args.n,len(tokens),len(tokenizer))
        print('p-value: ', pval)
        if pval < 0.01:
                logger.info('The watermark is detected')
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate text watermarked with a key')
    # parser.add_argument('--model',default='huggyllama/llama-7b',type=str,
    #         help='a HuggingFace model id of the model to generate from')
    parser.add_argument('--model',default='facebook/opt-1.3b',type=str,
            help='a HuggingFace model id of the model to generate from')
    parser.add_argument('--prompt',default='Watermarking is',type=str,
            help='an optional prompt for generation')
    parser.add_argument('--mode',default='bileve',type=str,
            help='watermarking mode')
    parser.add_argument('--gamma',default=0.001,type=float,
            help='weighted rank parameter')
    parser.add_argument('--m',default=512,type=int,
            help='the requested length of the generated text') # This demo uses 512-bit signature, so m should be larger than 512.
    parser.add_argument('--n',default=80,type=int,
            help='the length of the watermark key sequence')
    parser.add_argument('--d',default=44,type=int,
            help='use first d tokens as message')           
    parser.add_argument('--key',default=42,type=int,
            help='a key for generating the random watermark sequence')
    parser.add_argument('--seed',default=0,type=int,
            help='a seed for reproducibile randomness')
    parser.add_argument('--out_dir', default='./log_test',
        help='directory for saving results')
    parser.add_argument('--logfile', default='test.log',
                    help='directory for saving results')

    main(parser.parse_args())
