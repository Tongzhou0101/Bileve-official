import sys
import numpy as np
from mersenne import mersenne_rng
import pyximport
pyximport.install(reload_support=True, language_level=sys.version_info[0],
                  setup_args={'include_dirs':np.get_include()})
from levenshtein import levenshtein
from fastecdsa import ecdsa

def verify(logger, watermarked_tokens, public_key, tokenizer, m, d):
    # Extract the last m+d tokens from the watermarked text
    logger.info('Extraction...')

    msg = tokenizer.decode(watermarked_tokens[-m-d:-m])  # Decode the message tokens
    sig = watermarked_tokens[-m:]  # Extract signature tokens

     # Extract the signature bits
    signature_bits = []

    for i in range(m):
        res = sig[i] % 2
        signature_bits.append(res.item())

    r = sum(bit << i for i, bit in enumerate(reversed(signature_bits[:256])))
    s = sum(bit << i for i, bit in enumerate(reversed(signature_bits[256:512])))
    logger.info('r_extracted: %s', r)
    logger.info('s_extracted: %s', s)
    valid = ecdsa.verify((r, s), msg, public_key)
    

    return valid

def permutation_test(tokens,key,n,k,vocab_size,n_runs=100):
    rng = mersenne_rng(key)
    xi = np.array([rng.rand() for _ in range(n*vocab_size)], dtype=np.float32).reshape(n,vocab_size)
    test_result = detect(tokens,n,k,xi)

    p_val = 0
    for run in range(n_runs):
        xi_alternative = np.random.rand(n, vocab_size).astype(np.float32)
        null_result = detect(tokens,n,k,xi_alternative)

        # assuming lower test values indicate presence of watermark
        p_val += null_result <= test_result

    return (p_val+1.0)/(n_runs+1.0)


def detect(tokens,n,k,xi,gamma=0.0):
    m = len(tokens)
    n = len(xi)
    # print(m,k,n)
    A = np.empty((m-(k-1),n))
    for i in range(m-(k-1)):
        for j in range(n):
            A[i][j] = levenshtein(tokens[i:i+k],xi[(j+np.arange(k))%n],gamma)

    return np.min(A)