import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm.auto import tqdm
import regex as re

# def parse_text_into_versions(text):
#     """
#     Parse the text with <ERR> tags and return two versions: one with errors and one with corrections.
#
#     :param text: Input text containing <ERR> tags with target and incorrect words.
#     :return: A tuple containing two versions: (text_with_errors, text_with_corrections).
#     """
#     err_pattern = re.compile(r'<ERR targ=([^>]+)> ([^<]+) </ERR>')
#     text_with_errors = text
#     text_with_corrections = text
#
#     for match in err_pattern.finditer(text):
#         correct_word_target = match.group(1)  # The correct word in the targ attribute
#         incorrect_word = match.group(2)  # The actual incorrect word
#
#         # Replace in the error version (remove <ERR> tags and leave the incorrect word)
#         text_with_errors = text_with_errors.replace(match.group(0), incorrect_word)
#         # Replace in the correction version (replace incorrect word with the correct word)
#         text_with_corrections = text_with_corrections.replace(match.group(0), correct_word_target)
#
#     return text_with_errors, text_with_corrections
#
# def clean_and_split_file(file_content):
#     """
#     Clean the file content by removing lines that contain only a number with a dot and the following line,
#     then split the text by new lines.
#
#     :param file_content: The entire text content of the file.
#     :return: A list of lines (articles) split by new lines.
#     """
#     lines = file_content.split('\n')
#     cleaned_lines = []
#     skip_next_line = False
#
#     for i, line in enumerate(lines):
#         if skip_next_line:
#             skip_next_line = False
#             continue
#
#         if re.match(r'^\d+\.$', line.strip()):  # Matches lines that are numbers with a dot
#             skip_next_line = True  # Skip this line and mark the next line to be skipped
#         else:
#             cleaned_lines.append(line)
#
#     # Remove any empty lines
#     articles = [line.strip() for line in cleaned_lines if line.strip()]
#
#     return articles
#
# def process_file_into_versions(file_content):
#     """
#     Process the cleaned and split file into two lists: one with errors and one with corrections.
#
#     :param file_content: The entire text content of the file.
#     :return: Two lists: (list_with_errors, list_with_corrections).
#     """
#     articles = clean_and_split_file(file_content)
#
#     list_with_errors = []
#     list_with_corrections = []
#
#     for article in articles:
#         text_with_errors, text_with_corrections = parse_text_into_versions(article)
#         list_with_errors.append(text_with_errors)
#         list_with_corrections.append(text_with_corrections)
#
#     return list_with_errors, list_with_corrections

with open('data/entries.train', 'r') as file:
    file_content = file.read()

# Initialize lists to store sentences
sentences_with_errors = []
corrected_sentences = []

# Split the data into lines
lines = file_content.strip().split('\n')[:10000]

for line in lines:
    # Strip whitespace and skip empty lines
    line = line.strip()
    if not line:
        continue  # Skip blank lines

    # Split each line into columns using tab as the delimiter
    columns = line.split('\t')

    # Check if the first column is not '0'
    if columns[0] != '0':
        # The sentence with errors is in column 5
        sentence_with_error = columns[4]
        # The corrected sentence(s) are in columns 5 onwards
        # Join any additional columns in case the sentence contains tabs
        corrected_sentence = '\t'.join(columns[5:])

        sentences_with_errors.append(sentence_with_error)
        corrected_sentences.append(corrected_sentence)

# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')


# Create a custom dataset class
class SpellCorrectionDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer, max_length=256):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_prefix = "Correct grammar errors: "

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        source = self.input_texts[idx]
        target = self.target_texts[idx]

        source_encodings = self.tokenizer(
            self.task_prefix + source,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        target_encodings = self.tokenizer(
            target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        labels = target_encodings['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss

        return {
            'input_ids': source_encodings['input_ids'].flatten(),
            'attention_mask': source_encodings['attention_mask'].flatten(),
            'labels': labels.flatten(),
        }


# Create dataset and dataloader
dataset = SpellCorrectionDataset(sentences_with_errors, corrected_sentences, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set up optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 20  # Adjust the number of epochs as needed
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}')
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({'Loss': loss.item()})
    avg_loss = epoch_loss / len(dataloader)
    print(f'Average loss for epoch {epoch + 1}: {avg_loss}')

# Save the fine-tuned model
model.save_pretrained('t5_grammar_corrector')
tokenizer.save_pretrained('t5_grammar_corrector')


# Example usage
def correct_spelling(sentence):
    model.eval()
    input_ids = tokenizer.encode(f"Correct grammar errors: {sentence}", return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids)
    corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_sentence


# Test the model
test_sentence = "I took a medicine , so I feel better for now ."
corrected = correct_spelling(test_sentence)
print(f"Original: {test_sentence}")
print(f"Corrected: {corrected}")
