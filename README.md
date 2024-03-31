# BERT: Bidirectional Encoder Representation Transformer

BERT (Bidirectional Encoder Representation Transformer) is a transformer-based language model developed by the Google Research Team in 2018. It has gained significant attention and is widely applied in various Natural Language Processing (NLP) tasks due to its impressive performance.

## Overview

BERT is a pre-trained model that can be fine-tuned for a wide range of NLP tasks, including question-answering, text tagging, and sentence classification. It is based on the transformer architecture, which uses self-attention mechanisms to capture the contextual relationships between words in a sentence.

One of the key innovations of BERT is its bidirectional training approach. Unlike traditional language models that process text sequentially, BERT can capture the context from both directions (left-to-right and right-to-left) simultaneously, resulting in a more comprehensive understanding of the text.

## Files:
- **bert_database.py** : define the dataset to use for model and also train the tokenizer and save it.
- **embeddings.py** : as we know in bert we have three embedding, token, segment and position and we need to add all these three together to identify the final embedding, in this file we define that.
- **model.py**: developed the final BERT model as per the original paper
- **optimizer.py**: this is the scheduled optimizer wrap class.
- **tokinization_utils.py** : utility functions relating to tokenization.
- **utils.py** : other utility functions required to develop the model.
- **train.py**: final train loop code.

## License

This project is licensed under the [MIT License](LICENSE).
