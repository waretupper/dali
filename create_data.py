import random

n_train = 547334
n_dev = 1000

with open('dataset/emea-lex-wsub+it/test.1.tok.it.bpe.src') as fEn, \
     open('dataset/emea-lex-wsub+it/test.1.tok.it.bpe.tgt') as fDe:

    zipped = list(zip(fDe, fEn))

    shuffle = random.shuffle(zipped)

    dev = zipped[-n_dev:]
    train = zipped[:-n_dev][:n_train]
    de_dev, en_dev = zip(*dev)
    de_train, en_train = zip(*train)

with open('dataset/emea-lex-wsub+it/it-dev.bpe.en') as dev_it_en, \
     open('dataset/emea-lex-wsub+it/it-dev.bpe.de') as dev_it_de, \
     open('dataset/emea-lex-wsub+it/it-train.bpe.clean.en') as train_it_en, \
     open('dataset/emea-lex-wsub+it/it-train.bpe.clean.de') as train_it_de:


    with open("dataset/emea-lex-wsub+it/emea-dev.bpe.de", "w") as outfile:
        outfile.write("".join(de_dev))
    with open("dataset/emea-lex-wsub+it/emea-dev.bpe.en", "w") as outfile:
        outfile.write("".join(en_dev))

    with open("dataset/emea-lex-wsub+it/emea-wsub-unsup+it-para-dev.bpe.de", "w") as outfile:
        outfile.write("".join(de_dev))
        outfile.write("".join(list(dev_it_de)))
    with open("dataset/emea-lex-wsub+it/emea-wsub-unsup+it-para-dev.bpe.en", "w") as outfile:
        outfile.write("".join(en_dev))
        outfile.write("".join(list(dev_it_en)))


    with open("dataset/emea-lex-wsub+it/emea-train.bpe.de", "w") as outfile:
        outfile.write("".join(de_train))
    with open("dataset/emea-lex-wsub+it/emea-train.bpe.en", "w") as outfile:
        outfile.write("".join(en_train))

    with open("dataset/emea-lex-wsub+it/emea-wsub-unsup+it-para-train.bpe.de", "w") as outfile:
        outfile.write("".join(de_train))
        outfile.write("".join(list(train_it_de)))

    with open("dataset/emea-lex-wsub+it/emea-wsub-unsup+it-para-train.bpe.en", "w") as outfile:
        outfile.write("".join(en_train))
        outfile.write("".join(list(train_it_en)))
