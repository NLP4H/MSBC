import re
from nltk.tokenize import TreebankWordTokenizer

if __name__ == "__main__":
    input_text = "master_path/data/neurology_notes/processed_data/pre_training/notes_for_pretraining.txt"
    output_text = "master_path/data/neurology_notes/processed_data/pre_training/notes_for_pretraining_uncased_sentence_nltk.txt"
    
    with open(output_text, 'w') as out_txt:
        with open(input_text) as in_text:
            for line in in_text:
                value = line.strip().lower()
                value = re.sub(r'[\r\n]+', ' ', value)
                value = re.sub(r'[^\x00-\x7F]+', ' ', value)
                tokenized = TreebankWordTokenizer().tokenize(value)
                sentence = ' '.join(tokenized)
                sentence = re.sub(r"\s's\b", "'s", sentence)
                out_txt.write(sentence+' ')

