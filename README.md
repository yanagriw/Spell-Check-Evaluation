# Introduction

The goal of this project is to develop a **language model for context-based spelling correction** and compare its performance with well-known Python libraries such as SymSpell, LanguageTool, and TextBlob.

The result is a **fine-tuned T5 language model** trained on a **synthetically generated dataset** tailored for our specific task.

**Original:** *I wos in Franse lasst yer.*

**Corrected with fine-tuned T5 model:** *I was in France last year.*

# Datasets

## Birkbeck

The Birkbeck spelling error corpus is a dataset containing isolated words with spelling errors and their correct forms. It is commonly used for evaluating spell-checking algorithms at the word level, without considering contextual information.

## Holbrook

The Holbrook corpus is one of the few datasets that include real-world spelling errors within sentence contexts. Due to its relatively small size, we use it only for evaluation purposes. For training, we rely on a generated dataset to provide sufficient data.

## Gutenberg with Synthetically Generated Typos

We created a dataset based on texts from Project Gutenberg, where we synthetically introduced spelling errors. For each word in the text, with a probability of 30%, we randomly applied one of the following errors:

- **Swap character**: Two adjacent characters are swapped.
- **Delete character**: A character is removed.
- **Insert character**: A random character is inserted into the word.

## How to Download

To download the datasets `birkbeck.dat` and `holbrook.dat`, run the script `download_data.py`.

To download the Gutenberg dataset and generate synthetic errors, run the script `generate_data.py`.

All data will be saved in the newly created `data` directory.

# Word-Level Corrections

We tested the `symspell` library on the `birkbeck.dat` dataset.

SymSpell is a fast and efficient spell-checking library that uses a compressed dictionary and Levenshtein distance to find possible corrections for misspelled words.

To evaluate the results, run the script `check_words.py`.

We used metrics such as Edit Distance and Accuracy (Exact Match) and obtained the following results:

- **Average Edit Distance**: 2.32 (per word)
- **Accuracy**: 33.08%

### Limitations of SymSpell

- **Limited to word-level corrections**: Does not consider the context in which a word appears.
- **Cannot handle homophones or grammatical issues**: Fails to correct words that sound the same or address grammatical errors.
- **Lacks understanding of context or grammar**: Corrections are based solely on edit distance without semantic understanding.

For example:

- **Original word (assumed correct)**: *youth*
- **Word to check**: *uth*
- **Suggested correction**: *utah*
- **Edit Distance**: 3

The tool selects a word closest to the input based on edit distance, but in context, this correction is incorrect.

# Context-Based Corrections

## Fine-Tuning the T5 Language Model

To achieve **better spelling correction** that accounts for context, we fine-tuned the **T5 (Text-to-Text Transfer Transformer)** language model.

### About T5

T5 is a transformer-based language model developed by Google. It treats every NLP problem as a text-to-text task, enabling it to be fine-tuned for various applications like translation, summarization, and text correction.

By framing spelling correction as a text-to-text problem, we input sentences with errors and train the model to output the corrected sentences, **leveraging contextual information**.

### Training the Model

To start the training process, run the script `model_spell_checker.py`.

Training is performed on our synthetically generated dataset `gutenberg_sentences_with_typos.txt`, which contains approximately 100,000 sentences.

### Accessing the Trained Model

The trained model is available on Hugging Face for convenient use:

[https://huggingface.co/yanagriw/T5-spell-checker](https://huggingface.co/yanagriw/T5-spell-checker)

## Testing the Model

To directly test the model, run the script `spell_checker_T5.py`. Enter text with spelling errors into the console and press **Enter**. To exit, type `stop` and press **Enter**.

## Examples

- **Original:** *I coudnt go to sleap becuz the dog was barkin all nite.*
- **Corrected:** *I could not go to sleep because the dog was barking all night.*

- **Original:** *I tryed to bild a sandcastle but the wavs kept knockin it down.*
- **Corrected:** *I tried to build a sandcastle but the waves kept knocking it down.*

## Performance Evaluation

For performance evaluation and comparison with other libraries like `language_tool_python` and `TextBlob`, run the script `check_text.py`. We use the BLEU (Bilingual Evaluation Understudy) metric on the `holbrook.dat` dataset since it contains real examples.

### About BLEU

BLEU is a metric for evaluating the quality of text which has been machine-translated from one language to another. It calculates the similarity between the machine's output and one or more reference translations, using a modified form of precision.

### Results

- **TextBlob** - BLEU: 69.11
- **LanguageTool** - BLEU: 73.68
- **T5 Model** - BLEU: 77.64

### About LanguageTool

`language_tool_python` is a Python wrapper for LanguageTool, an open-source proofreading software. It provides grammar, style, and spell checking, and can detect complex errors by considering the context of words within a sentence.

### About TextBlob

TextBlob is a Python library for processing textual data. It offers simple APIs for common NLP tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, and spelling correction. Its spell checker uses a probabilistic approach at the word level.

# Results

Our fine-tuned T5 model outperforms other methods in BLEU score, indicating superior correction quality. By leveraging context, the T5 model can accurately correct spelling errors that word-level tools often miss, highlighting the importance of contextual understanding in natural language processing applications.
