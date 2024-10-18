from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm.auto import tqdm
import torch


# Create a custom dataset class
class SpellCorrectionDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer, max_length=128):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_prefix = "Spell checker: "

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


def load_data(file_path, train_size=95000, eval_split=True):
    """
    Loads and processes data from a specified file.

    :param file_path: Path to the data file.
    :param max_lines: Number of lines for the training split.
    :param eval_split: If True, returns both training and evaluation splits.
    :return: Two lists containing sentences with errors and their corrected versions.
             Optionally, returns evaluation data.
    """
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Split the data into lines
    lines = file_content.strip().split('\n')

    # Initialize lists to store sentences
    sentences_with_errors = []
    corrected_sentences = []

    for line in lines:
        # Strip whitespace and skip empty lines
        line = line.strip()
        if not line:
            continue

        # Split each line into columns using tab as the delimiter
        columns = line.split('\t')
        sentence_with_error = columns[1]
        corrected_sentence = columns[0]

        sentences_with_errors.append(sentence_with_error)
        corrected_sentences.append(corrected_sentence)

    # Split the data into training and evaluation sets
    if eval_split:
        train_sentences_with_errors = sentences_with_errors[:train_size]
        train_corrected_sentences = corrected_sentences[:train_size]
        eval_sentences_with_errors = sentences_with_errors[train_size:]
        eval_corrected_sentences = corrected_sentences[train_size:]
        return train_sentences_with_errors, train_corrected_sentences, eval_sentences_with_errors, eval_corrected_sentences
    else:
        return sentences_with_errors, corrected_sentences


def main():
    # Initialize the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    # Load training and evaluation data
    train_sentences_with_errors, train_corrected_sentences, eval_sentences_with_errors, eval_corrected_sentences = load_data("data/gutenberg_sentences_with_typos.txt")

    # Create datasets and dataloaders
    train_dataset = SpellCorrectionDataset(train_sentences_with_errors, train_corrected_sentences, tokenizer)
    eval_dataset = SpellCorrectionDataset(eval_sentences_with_errors, eval_corrected_sentences, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}')
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
        avg_train_loss = epoch_loss / len(train_dataloader)
        print(f'Average training loss for epoch {epoch + 1}: {avg_train_loss}')

        # Evaluation loop
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc=f'Evaluating Epoch {epoch + 1}'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                eval_loss += loss.item()

        avg_eval_loss = eval_loss / len(eval_dataloader)
        print(f'Average evaluation loss for epoch {epoch + 1}: {avg_eval_loss}')

    # Save the fine-tuned model
    model.save_pretrained('t5_spell_corrector')
    tokenizer.save_pretrained('t5_spell_corrector')

if __name__ == '__main__':
    main()