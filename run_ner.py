from flair.data import NLPTaskDataFetcher, TaggedCorpus, NLPTask
from flair.embeddings import TextEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings, CharacterEmbeddings
from typing import List
import torch
import argparse

def load_data(train_file_path, dev_file_path, test_file_path):
    # get training, test and dev data
    sentences_train: List[Sentence] = NLPTaskDataFetcher.read_conll_sequence_labeling_data(train_file_path)
    sentences_dev: List[Sentence] = NLPTaskDataFetcher.read_conll_sequence_labeling_data(dev_file_path)
    sentences_test: List[Sentence] = NLPTaskDataFetcher.read_conll_sequence_labeling_data(test_file_path)

    # return corpus
    return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

def train(train_file_path, dev_file_path, test_file_path):
    # 1. get the corpus
    corpus: TaggedCorpus = load_data(train_file_path, dev_file_path, test_file_path)
    corpus.train = [sentence for sentence in corpus.train if len(sentence) > 0]
    corpus.test = [sentence for sentence in corpus.test if len(sentence) > 0]
    corpus.dev = [sentence for sentence in corpus.dev if len(sentence) > 0]
    print('corpus')
    print(corpus)

    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print('tag_dictionary')
    print(tag_dictionary.idx2item)

    # initialize embeddings
    embedding_types: List[TextEmbeddings] = [

        # GloVe embeddings
        WordEmbeddings('glove')
        ,
        # contextual string embeddings, forward
        CharLMEmbeddings('news-forward')
        ,
        # contextual string embeddings, backward
        CharLMEmbeddings('news-backward')
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # initialize sequence tagger
    from flair.tagging_model import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    if torch.cuda.is_available():
        tagger = tagger.cuda()

    # initialize trainer
    from flair.trainer import TagTrain

    trainer: TagTrain = TagTrain(tagger, corpus, test_mode=False)

    trainer.train('resources/taggers/example-ner', mini_batch_size=32, max_epochs=150, save_model=True,
                  train_with_dev=True, anneal_mode=True)

def main():
    parser = argparse.ArgumentParser(description='Run NER')
    parser.add_argument('--train', required=True)  # "data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev', required=True)  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test', required=True)  # "data/POS-penn/wsj/split1/wsj1.test.original"

    args = parser.parse_args()
    train_file_path = args.train
    dev_file_path = args.dev
    test_file_path = args.test

    train(train_file_path, dev_file_path, test_file_path)

if __name__ == '__main__':
    main()

