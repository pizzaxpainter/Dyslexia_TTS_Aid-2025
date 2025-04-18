import nltk
from nltk.translate import bleu_score
from collections import Counter

nltk.download('punkt')

def get_unigrams(sentence):
    return set(nltk.word_tokenize(sentence))

def calculate_sari(predictions, references):
    assert len(predictions) == len(references), "Predictions and references must have the same length."
    
    sari_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = get_unigrams(pred)
        ref_tokens = get_unigrams(ref)
        
        # Calculate Additions, Deletions, Substitutions
        added = pred_tokens - ref_tokens
        deleted = ref_tokens - pred_tokens
        substituted = pred_tokens & ref_tokens

        # Count occurrences of these tokens
        added_count = len(added)
        deleted_count = len(deleted)
        substituted_count = len(substituted)
        
        # Calculate precision, recall, and F1 score for each of the three aspects
        # Precision, Recall, F1 for Additions, Deletions, Substitutions
        precision = added_count / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = added_count / len(ref_tokens) if len(ref_tokens) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        sari_scores.append(f1)
    
    # Return the average SARI score
    return sum(sari_scores) / len(sari_scores)

# Example usage:
predictions = [
    "The cat sat on the mat.",
    "He likes apples."
]

references = [
    "A cat was on a mat.",
    "He enjoys eating apples."
]

sari_score = calculate_sari(predictions, references)
print(f"SARI score: {sari_score}")
