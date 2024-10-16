# Spell Check Evaluation (for Improving Writing Assistance Project)

### Traditional Spell-Checking Libraries - SymSpell

Limited to word-level corrections (no context sensitivity), cannot handle homophones or grammatical issues.

Lacks understanding of context or grammar, purely focused on edit distance.

Fast, highly efficient, dictionary-based, High performance, large dictionaries.

### Context Awareness

TextBlob - BLEU: 69.44, Edit Distance: 6.78

Ideal for simple context-aware corrections in text along with other NLP tasks. 
It uses an internal machine learning model to handle corrections.
Works well for general applications but is not as advanced as specialized grammar tools.

LanguageTool - BLEU: 72.78, Edit Distance: 6.89

Ideal for correcting grammar, style, and spelling with context-awareness. 
It can be used for both single sentences and entire documents.
Detects complex issues such as subject-verb agreement and contextual word misuse.
Highly effective for handling real-world text.