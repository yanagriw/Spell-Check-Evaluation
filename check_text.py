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


def parse_text_into_versions(text):
    """
    Parse the text with <ERR> tags and return two versions: one with errors and one with corrections.

    :param text: Input text containing <ERR> tags with target and incorrect words.
    :return: A tuple containing two versions: (text_with_errors, text_with_corrections).
    """
    err_pattern = re.compile(r'<ERR targ=([^>]+)> ([^<]+) </ERR>')
    text_with_errors = text
    text_with_corrections = text

    for match in err_pattern.finditer(text):
        correct_word_target = match.group(1)  # The correct word in the targ attribute
        incorrect_word = match.group(2)  # The actual incorrect word

        # Replace in the error version (remove <ERR> tags and leave the incorrect word)
        text_with_errors = text_with_errors.replace(match.group(0), incorrect_word)
        # Replace in the correction version (replace incorrect word with the correct word)
        text_with_corrections = text_with_corrections.replace(match.group(0), correct_word_target)

    return text_with_errors, text_with_corrections


def clean_and_split_file(file_content):
    """
    Clean the file content by removing lines that contain only a number with a dot and the following line,
    then split the text by blank lines.

    :param file_content: The entire text content of the file.
    :return: A list of articles/paragraphs split by blank lines.
    """
    lines = file_content.split('\n')
    cleaned_lines = []
    skip_next_line = False

    for i, line in enumerate(lines):
        if skip_next_line:
            skip_next_line = False
            continue

        if re.match(r'^\d+\.$', line.strip()):  # Matches lines that are numbers with a dot
            skip_next_line = True  # Mark the next line to be skipped
        else:
            cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    articles = [article.strip() for article in cleaned_text.split('\n\n') if article.strip()]

    return articles


def process_file_into_versions(file_content):
    """
    Process the cleaned and split file into two lists: one with errors and one with corrections.

    :param file_content: The entire text content of the file.
    :return: Two lists: (list_with_errors, list_with_corrections).
    """
    articles = clean_and_split_file(file_content)

    list_with_errors = []
    list_with_corrections = []

    for article in articles:
        text_with_errors, text_with_corrections = parse_text_into_versions(article)
        list_with_errors.append(text_with_errors)
        list_with_corrections.append(text_with_corrections)

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
