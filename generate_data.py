import nltk
import random
import os

nltk.download('wordnet')
from nltk.corpus import wordnet


def introduce_typo(word):
    """
    Introduces a typo in the given word. The typo can be a character swap, deletion, or insertion.

    :param word: The word to which a typo will be introduced.
    :return: The word with an introduced typo, or the original word if it's too short.
    """
    if len(word) < 3:
        return word

    typo_type = random.choice(['swap', 'delete', 'insert'])

    if typo_type == 'swap':  # Swap adjacent characters
        idx = random.randint(0, len(word) - 2)
        return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]

    elif typo_type == 'delete':  # Delete a character
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx + 1:]

    elif typo_type == 'insert':  # Insert a random character
        idx = random.randint(0, len(word))
        random_char = random.choice('abcdefghijklmnopqrstuvwxyz')
        return word[:idx] + random_char + word[idx:]

    return word

def extract_wordnet_words():
    """
    Extracts words from the WordNet corpus, replacing underscores with spaces in multi-word lemmas.

    :return: A set of words extracted from WordNet with underscores replaced by spaces and all words in lowercase.
    """
    words = set()
    for synset in wordnet.all_synsets():
        for lemma in synset.lemmas():
            word_with_spaces = lemma.name().replace('_', ' ')  # Replace underscores with spaces
            words.add(word_with_spaces.lower())  # Convert to lowercase and add to set
    return words

def save_words_with_typos(words, output_file):
    """
    Saves the original words and their typo-introduced counterparts in the specified format to a file.

    :param words: A set of words extracted from WordNet.
    :param output_file: The path to the file where the words and their typos will be saved.
    """
    with open(output_file, 'w') as f:
        for word in words:
            typo_word = introduce_typo(word)  # Introduce typo to the word
            f.write(f"${word}\n{typo_word}\n")  # Write original and typo words to file

def main():
    output_dir = "data"
    output_file = os.path.join(output_dir, "wordnet_with_typos.txt")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract words from WordNet
    word_list = extract_wordnet_words()

    # Save words with introduced typos
    save_words_with_typos(word_list, output_file)
    print(f"Saved corpus with typos to {output_file}")

if __name__ == "__main__":
    main()
