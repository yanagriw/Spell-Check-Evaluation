from symspellpy.symspellpy import SymSpell, Verbosity
import os
import nltk

def initialize_symspell(max_edit_distance_dictionary=2, prefix_length=7):
    """
    Initializes the SymSpell object and loads the dictionary.

    :param max_edit_distance_dictionary: Maximum allowed edit distance.
    :param prefix_length: Length of word prefix for dictionary lookup.
    :return: Initialized SymSpell object.
    """
    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)

    # Load the frequency dictionary
    dictionary_path = os.path.join('data/frequency_dictionary_en_82_765.txt')  # Download this file from https://github.com/mammothb/symspellpy
    if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
        print("Dictionary file not found.")
        return None

    return sym_spell

def check_word_symspell(word, sym_spell):
    """
    Checks the word using SymSpell and suggests corrections if necessary.

    :param word: The word to check.
    :param sym_spell: The SymSpell object.
    :return: Suggestion for the word.
    """
    suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)

    if suggestions:
        return suggestions[0].term  # Return the best suggestion
    return word  # Return the original word if no suggestions found

def check_words_in_file(file_path, sym_spell):
    """
    Reads the file and checks the words using SymSpell.

    :param file_path: Path to the file containing words.
    :param sym_spell: Initialized SymSpell object.
    """
    total_edit_distance = 0
    total_words = 0
    correct_suggestions = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip()

            # Skip lines starting with '$' (correct words)
            if line.startswith('$'):
                original_word = line[1:]  # Extract correct word without '$'
                print(f"Original word (assumed correct): {original_word}")
            else:
                typo_word = line
                print(f"Word to check: {typo_word}")

                # Check word using SymSpell
                suggestion = check_word_symspell(typo_word, sym_spell)
                print(f"Suggested correction: {suggestion}")

                # Calculate edit distance
                distance = nltk.edit_distance(suggestion, original_word)
                total_edit_distance += distance
                total_words += 1

                # Count correct suggestions (edit distance 0 means it's exactly correct)
                if distance == 0:
                    correct_suggestions += 1

                print(f"Edit Distance: {distance}")
                print('-' * 40)

    # Calculate and print average edit distance and accuracy
    if total_words > 0:
        avg_edit_distance = total_edit_distance / total_words
        accuracy = correct_suggestions / total_words * 100
        print(f"Average Edit Distance: {avg_edit_distance}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("No words to evaluate.")


def main():
    sym_spell = initialize_symspell()

    if sym_spell:
        # Path to the input file
        file_path = 'data/aspell.dat'

        check_words_in_file(file_path, sym_spell)

if __name__ == "__main__":
    main()
