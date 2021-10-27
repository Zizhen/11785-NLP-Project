#!/bin/bash

mkdir -p examples/glove
curl -Lo glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip -d examples/glove
rm glove.840B.300d.zip


mkdir -p examples/fasttext
curl -Lo crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip -d examples/fasttext
rm crawl-300d-2M.vec.zip


mkdir -p examples/benchmark-result
