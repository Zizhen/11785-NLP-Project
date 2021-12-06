# 11785-NLP-Project
This project aims to generate high-level ensemble embeddings by implementing different fusing methods. We have proposed several strategies to ensemble most recent RoBERTa
model and traditional models like LSTM and GloVe, hoping to combine the strength of all models and achieve overall better performance than individual models. Experiments on different ensembling combinations have shown that concatenating LSTM and GloVe pretrained word embeddings and using a fine tuned linear layer with a shape that’s half of input size to further extract features has given promising results in a wide range of NLP benchmarks. We have achieved percentage accuracy increase in almost every classification tasks and linguistic probing tasks.

./setup-glove-fasttext.sh

./set-env.sh

cd examples

python bow.py
