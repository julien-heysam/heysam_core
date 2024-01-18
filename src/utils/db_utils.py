import logging

import src

from sentence_transformers import SentenceTransformer

logging.info("This is an info message from some_module.")

embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
info_model = {
    "dimension": embeddings.get_sentence_embedding_dimension(),
    "max_seq_length": embeddings.max_seq_length,
}

if __name__ == '__main__':
    # Call your function or other code
    logging.info("This is an info message from some_module.")
