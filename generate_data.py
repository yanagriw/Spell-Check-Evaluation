import nltk
import random
import os

nltk.download('gutenberg')
nltk.download('punkt_tab')
from nltk.corpus import gutenberg

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
        random_char = random.choice('abcdefghijklmnopqrstuvwxyz     ')
        return word[:idx] + random_char + word[idx:]

    return word

def extract_sentences():
    """
    Extracts sentences from the Gutenberg corpus.

    :return: A list of sentences, where each sentence is a list of words.
    """
    sentences = gutenberg.sents()
    return sentences

def save_sentences_with_typos(sentences, output_file, typo_probability=0.3):
    """
    Saves original sentences and their typo-introduced counterparts to a file, separated by a tab.

    :param sentences: A list of sentences, where each sentence is a list of words.
    :param output_file: The path to the file where the sentences will be saved.
    :param typo_probability: The probability of introducing a typo into each word.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            original_sentence = ' '.join(sentence)

            # Process the sentence to introduce typos
            typo_sentence_words = []
            for word in sentence:
                if random.random() < typo_probability:
                    new_word = introduce_typo(word)
                else:
                    new_word = word
                typo_sentence_words.append(new_word)

            typo_sentence = ' '.join(typo_sentence_words)

            # Write the original and typo sentence separated by a tab
            f.write(f"{original_sentence}\t{typo_sentence}\n")

def main():
    output_dir = "data"
    output_file_sentences = os.path.join(output_dir, "gutenberg_sentences_with_typos.txt")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract sentences from Gutenberg
    sentences = extract_sentences()

    save_sentences_with_typos(sentences, output_file_sentences)
    print(f"Saved corpus with typos to {output_file_sentences}")

if __name__ == "__main__":
    main()