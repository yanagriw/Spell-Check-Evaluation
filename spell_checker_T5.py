from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def correct_spelling(input_text, model, tokenizer, device):
    # Tokenize the input text
    inputs = tokenizer("Spell checker: " + input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)

    # Decode the output
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def main():
    # Load the fine-tuned model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained("yanagriw/T5-spell-checker")
    model = T5ForConditionalGeneration.from_pretrained("yanagriw/T5-spell-checker")

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Enter sentences to correct (type 'stop' to exit):")
    while True:
        # Read input from the user
        input_sentence = input(">> ")
        if input_sentence.strip().lower() == 'stop':
            break

        # Correct the spelling
        corrected_sentence = correct_spelling(input_sentence, model, tokenizer, device)

        # Display the result
        print("Corrected:", corrected_sentence)

if __name__ == '__main__':
    main()
