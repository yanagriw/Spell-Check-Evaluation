import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm.auto import tqdm


with open('data/gutenberg_sentences_with_typos.txt', 'r') as file:
    file_content = file.read()

# Initialize lists to store sentences
sentences_with_errors = []
corrected_sentences = []

# Split the data into lines
lines = file_content.strip().split('\n')

for line in lines:
    # Strip whitespace and skip empty lines
    line = line.strip()
    if not line:
        continue  # Skip blank lines

    # Split each line into columns using tab as the delimiter
    columns = line.split('\t')

    # The sentence with errors is in column 5
    sentence_with_error = columns[1]
    # The corrected sentence(s) are in columns 5 onwards
    # Join any additional columns in case the sentence contains tabs
    corrected_sentence = columns[0]

    sentences_with_errors.append(sentence_with_error)
    corrected_sentences.append(corrected_sentence)

# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')


# Create a custom dataset class
class SpellCorrectionDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer, max_length=512):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_prefix = "Correct typos: "

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
epochs = 10  # Adjust the number of epochs as needed
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
model.save_pretrained('t5_spell_corrector')
tokenizer.save_pretrained('t5_spell_corrector')


# Example usage
def correct_spelling(sentence):
    model.eval()
    input_ids = tokenizer.encode(f"Correct typos: {sentence}", return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids)
    corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_sentence


# Test the model
test_sentence = "If you want to her how the word is sad, we can assist with that to."
corrected = correct_spelling(test_sentence)
print(f"Original: {test_sentence}")
print(f"Corrected: {corrected}")
