import random
import pickle

import openai

from llama_index.llms import OpenAI
from llama_index.schema import BaseNode
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SentenceSplitter
from llama_index.evaluation import generate_question_context_pairs, EmbeddingQAFinetuneDataset

from src import PROJECT_API_KEYS, PROJECT_PATHS

openai.api_key = PROJECT_API_KEYS.OPENAI_API_KEY

DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
# You are:
A data-scientist expert in data-generation, \
You are building a dataset for retrieval evaluation.

# Your task:
Given the context information below and not prior knowledge. \
Generate {num_questions_per_chunk} question(s) from the provided context. \
Restrict the questions to the context information provided. \
Please ignore any kind of `coding/programming` question.

# Output format
If you can't find any relevant question(s), please answer with `idk`.

# Context information is below.
-------------
{context_str}
-------------
"""


class GenerateRetrievalDataset:

    def __init__(self, model_name: str = "gpt-4") -> None:
        self.llm = OpenAI(model=model_name)

    def get_nodes(self, input_dir: str, chunk_size: int):
        documents = SimpleDirectoryReader(input_dir=input_dir, recursive=True).load_data()
        node_parser = SentenceSplitter(chunk_size=chunk_size)
        nodes = node_parser.get_nodes_from_documents(documents)
        for idx, node in enumerate(nodes):
            node.id_ = f"node_{idx}"

        return nodes

    def save_nodes(self, nodes: list[BaseNode], path: str):
        with open(path, 'wb') as file:
            pickle.dump(nodes, file)

    def save_dataset(self, qa_dataset: EmbeddingQAFinetuneDataset, path: str):
        qa_dataset.save_json(path)

    def load_dataset(self, path: str) -> EmbeddingQAFinetuneDataset:
        return EmbeddingQAFinetuneDataset.from_json(path)

    def build_dataset(self, nodes: list[BaseNode], data_size: int = None, prompt: str = DEFAULT_QA_GENERATE_PROMPT_TMPL) -> EmbeddingQAFinetuneDataset:
        nodes = random.choices(nodes, k=data_size) if data_size else nodes
        qa_dataset = generate_question_context_pairs(nodes, llm=self.llm, num_questions_per_chunk=1, qa_generate_prompt_tmpl=prompt)

        return qa_dataset


if __name__ == '__main__':
    gen = GenerateRetrievalDataset()
    nodes = gen.get_nodes(input_dir="/Users/julienwuthrich/GitHub/heysam/orgs/split/artifacts/passages/cleaned/", chunk_size=512)
    gen.save_nodes(nodes, path=PROJECT_PATHS.INTERIM_DATA / "512_tokens" / "nodes.pickle")
    qa_dataset = gen.build_dataset(nodes=nodes, data_size=512)
    gen.save_dataset(qa_dataset=qa_dataset, path=PROJECT_PATHS.INTERIM_DATA / "512_tokens" / "qa_dataset.json")
