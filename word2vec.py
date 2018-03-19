import gensim
import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def extract_questions():
    """
    Extract questions for making word2vec model.
    """
    df1 = pd.read_csv("./data/train.csv")
    df2 = pd.read_csv("./data/test.csv")

    for dataset in [df1, df2]:
        for i, row in dataset.iterrows():
            if i != 0 and i % 1000 == 0:
                logging.info("read {0} sentences".format(i))

            if row['question1']:
                yield gensim.utils.simple_preprocess(row['question1'])
            if row['question2']:
                yield gensim.utils.simple_preprocess(row['question2'])


documents = list(extract_questions())
logging.info("Done reading data file")

model = gensim.models.Word2Vec(documents, size=300)
model.train(documents, total_examples=len(documents), epochs=10)
model.save("./data/Quora-Question-Pairs.w2v")
