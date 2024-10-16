import re
from textblob import TextBlob
import language_tool_python
import sacrebleu
import nltk
from tqdm import tqdm

# Initialize LanguageTool for English
tool = language_tool_python.LanguageTool('en-US')

def compute_bleu(candidate_texts, reference_texts):
    """
    Compute BLEU score between the candidate texts and reference texts.

    :param candidate_texts: List of corrected texts
    :param reference_texts: List of reference texts (ground truth)
    :return: BLEU score
    """
    bleu = sacrebleu.corpus_bleu(candidate_texts, [reference_texts])
    return bleu.score


def compute_edit_distance(candidate_texts, reference_texts):
    """
    Compute the average edit distance between the candidate texts and reference texts.

    :param candidate_texts: List of corrected texts
    :param reference_texts: List of reference texts (ground truth)
    :return: Average edit distance
    """
    total_distance = 0
    num_examples = len(candidate_texts)

    for i in range(num_examples):
        candidate = candidate_texts[i]
        reference = reference_texts[i]
        distance = nltk.edit_distance(candidate, reference)
        total_distance += distance

    average_distance = total_distance / num_examples
    return average_distance


def parse_line_into_versions(line):
    """
    Parse a line with <ERR> tags and return two versions: one with errors and one with corrections.

    :param line: Input line containing <ERR> tags with target and incorrect words.
    :return: A tuple containing two versions: (line_with_errors, line_with_corrections).
    """
    err_pattern = re.compile(r'<ERR targ=([^>]+)> ([^<]+) </ERR>')
    line_with_errors = line
    line_with_corrections = line

    for match in err_pattern.finditer(line):
        correct_word_target = match.group(1)  # The correct word in the targ attribute
        incorrect_word = match.group(2)  # The actual incorrect word

        # Replace in the error version (remove <ERR> tags and leave the incorrect word)
        line_with_errors = line_with_errors.replace(match.group(0), incorrect_word)
        # Replace in the correction version (replace the entire <ERR> tag with the correct word)
        line_with_corrections = line_with_corrections.replace(match.group(0), correct_word_target)

    # Remove any remaining <ERR> tags (if any)
    line_with_errors = re.sub(r'</?ERR[^>]*>', '', line_with_errors)
    line_with_corrections = re.sub(r'</?ERR[^>]*>', '', line_with_corrections)

    return line_with_errors.strip(), line_with_corrections.strip()


def process_file_into_versions(file_content):
    """
    Process the file content line by line and return lists of sentences with errors and corrections.

    :param file_content: The entire text content of the file.
    :return: Two lists: (list_with_errors, list_with_corrections).
    """
    lines = file_content.strip().split('\n')

    list_with_errors = []
    list_with_corrections = []

    for line in lines:
        line = line.strip()

        if '<ERR' in line:
            # Process lines that contain <ERR> tags
            line_with_errors, line_with_corrections = parse_line_into_versions(line)
            list_with_errors.append(line_with_errors)
            list_with_corrections.append(line_with_corrections)

    return list_with_errors, list_with_corrections


def correct_with_textblob(text):
    """
    Correct text using TextBlob.

    :param text: Input text with errors
    :return: Corrected text
    """
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text


def correct_with_languagetool(text):
    """
    Correct text using LanguageTool.

    :param text: Input text with errors
    :return: Corrected text
    """
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text


def evaluate_and_save_corrections(errors_list, corrections_list):
    """
    Apply TextBlob and LanguageTool corrections, compute metrics, and save the corrections.
    Display a progress bar using tqdm.

    :param errors_list: List of texts with errors
    :param corrections_list: List of correct texts
    """
    textblob_corrections = []
    languagetool_corrections = []

    # Using tqdm for the progress bar
    for text_with_errors in tqdm(errors_list, desc="Correcting texts"):
        # Correct using TextBlob
        tb_corrected = correct_with_textblob(text_with_errors)
        textblob_corrections.append(tb_corrected)

        # Correct using LanguageTool
        lt_corrected = correct_with_languagetool(text_with_errors)
        languagetool_corrections.append(lt_corrected)


    print("Computing metrics for TextBlob...")
    # Compute BLEU and Edit Distance for TextBlob corrections
    textblob_bleu = compute_bleu(textblob_corrections, corrections_list)
    textblob_edit_distance = compute_edit_distance(textblob_corrections, corrections_list)

    print("Computing metrics for LanguageTool...")
    # Compute BLEU and Edit Distance for LanguageTool corrections
    languagetool_bleu = compute_bleu(languagetool_corrections, corrections_list)
    languagetool_edit_distance = compute_edit_distance(languagetool_corrections, corrections_list)

    # Print metrics
    print(f"TextBlob - BLEU: {textblob_bleu:.2f}, Edit Distance: {textblob_edit_distance:.2f}")
    print(f"LanguageTool - BLEU: {languagetool_bleu:.2f}, Edit Distance: {languagetool_edit_distance:.2f}")

    # Save corrections to files
    with open('textblob_corrections.txt', 'w') as tb_file, open('languagetool_corrections.txt', 'w') as lt_file:
        for tb_corr, lt_corr in zip(textblob_corrections, languagetool_corrections):
            tb_file.write(tb_corr + '\n\n')
            lt_file.write(lt_corr + '\n\n')

def main():

    with open('data/holbrook.dat', 'r') as file:
        file_content = file.read()

    # Process the file to get error texts and reference corrections
    errors_list, corrections_list = process_file_into_versions(file_content)

    # Evaluate corrections and save the results
    evaluate_and_save_corrections(errors_list, corrections_list)


if __name__ == "__main__":
    main()
