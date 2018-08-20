"""
Input: a conll03 file/sents file/or the file that compatible with flair.
Output: a conll03 file which shows the identified information types.
"""
from flair.data import Sentence
from flair.models.sequence_tagger_model import SequenceTagger
from oppnlp.utils import read_sents_file

import fire
from pathlib import Path
from typing import List

def load_data(sents_file_path):
    """Input: a file with 1 sentence per line.
    Output: a list of sentences."""
    sents_file_path = Path(sents_file_path)
    lines = sents_file_path.read_text().split('\n')
    return [Sentence(line.strip()) for line in lines]

def write_tagged_sents(output_file_path, tagged_sents):
    conll03_sents : List[str] = []
    for sent in tagged_sents:
        conll03_str = []
        for tok in sent.tokens:
            # Default pos and chunk tag are O (other) which are unused.
            conll03_str.append('{} O O {}'.format(tok.text, tok.get_tag(tag_type='ner')))
        conll03_sents.append('\n'.join(conll03_str))

    output_file_path.write_text('\n'.append(conll03_sents))

def perform_ner(model_file_path, sents_file_path):
    # Load the pre-trained model from disk.
    tagger : SequenceTagger = SequenceTagger.load_from_file(model_file_path)

    # Make prediction.
    sentences = read_sents_file(sents_file_path)
    tagged_sentences = tagger.predict(sentences)

    # Write output.
    sents_file_path = Path(sents_file_path)
    output_file_path = sents_file_path.with_suffix('.ner.conll03')
    write_tagged_sents(output_file_path, tagged_sentences)
    return output_file_path

def default_perform_ner(sents_file_path):
    model_file_path = Path(__file__).parent.parent / 'resources/taggers/1p_col_3p_share/final-model.pt'
    return perform_ner(model_file_path, sents_file_path)

if __name__ == '__main__':
    fire.Fire(perform_ner)

