from datasets import load_dataset
from collections import Counter

def tokenize_with_vocab(
    sentence: str,
    vocabulary: list[str],
    lowercase: bool = True,
) -> list[str]:
    """
    Tokenize a sentence into vocabulary tokens (words, subwords, characters,
    punctuation, ...) in the order they appear, using greedy longest-match.

    Characters that cannot be covered by any vocabulary entry are skipped.

    Args:
        sentence: The input text.
        vocabulary: Allowed tokens (any strings, not just words).
        lowercase: Lowercase the sentence first (only if your vocab is lowercase).

    Returns:
        The list of matched tokens, in sentence order.
    """
    vocab_set = set(vocabulary)
    if not vocab_set:
        return []

    # No candidate can be longer than the longest vocab entry
    max_len = max(len(t) for t in vocab_set)

    if lowercase:
        sentence = sentence.lower()

    tokens = []
    i, n = 0, len(sentence)
    while i < n:
        # Try the longest possible match first, then shrink
        for j in range(min(n, i + max_len), i, -1):
            piece = sentence[i:j]
            if piece in vocab_set:
                tokens.append(piece)
                i = j
                break
        else:
            i += 1  # nothing matches here: skip one character
    return tokens

def bag_of_words(sentence: str, vocabulary: list[str]) -> list[int]:
    """
    Convert a sentence into its bag-of-words vector representation
    based on the given vocabulary.

    Args:
        sentence: The input text string.
        vocabulary: A list of unique words defining the feature space.

    Returns:
        A list of counts where each index corresponds to the
        count of the matching vocabulary word in the sentence.
    """
    # Normalize: lowercase and split into tokens
    tokens = tokenize_with_vocab(sentence, vocabulary)
    counts = Counter(tokens)
    return [counts[t] for t in vocabulary]

# Load the SNLI dataset
dataset = load_dataset("stanfordnlp/snli")

# Get the first example from the train partition
first_example = dataset['train'][0]

# The 'premise' is the first sentence (hypothesis is the second)
first_sentence = first_example['premise']

# Define a simple vocabulary for testing
# In practice, you might build this from the entire corpus or use a pre-trained embedding
vocabulary = list(set(first_sentence.lower().split()))

vector = bag_of_words(first_sentence, vocabulary)

print(f"First Sentence: {first_sentence}")
print(f"Vocabulary: {vocabulary}")
print(f"Bag-of-Words Vector: {vector}")
