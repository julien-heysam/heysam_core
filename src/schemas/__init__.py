from llama_index.evaluation import RetrieverEvaluator
from llama_index.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)
from llama_index.evaluation import generate_question_context_pairs
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.node_parser import SentenceSplitter
from llama_index.llms import OpenAI
from llama_index.evaluation import RetrieverEvaluator

import asyncio
from tqdm import tqdm
import openai
from llama_index.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)

openai.api_key = "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"
# documents = SimpleDirectoryReader(input_dir="/Users/julienwuthrich/GitHub/heysam/orgs/split/artifacts/passages/cleaned/", recursive=True,).load_data()

# node_parser = SentenceSplitter(chunk_size=4000)
# nodes = node_parser.get_nodes_from_documents(documents)
# for idx, node in enumerate(nodes):
#     node.id_ = f"node_{idx}"

# llm = OpenAI(model="gpt-4")
# service_context = ServiceContext.from_defaults(llm=llm)

# vector_index = VectorStoreIndex(nodes, service_context=service_context)
# vector_index.storage_context.persist(persist_dir="/Users/julienwuthrich/GitHub/heysam_core/notebooks/")


from llama_index import StorageContext, load_index_from_storage

# # rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="/Users/julienwuthrich/GitHub/heysam_core/notebooks/")

# # load index
vector_index = load_index_from_storage(storage_context)

retriever = vector_index.as_retriever(similarity_top_k=2)

metrics = ["mrr", "hit_rate"]
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    metrics, retriever=retriever
)

import pandas as pd


def display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()
    columns = {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}
    metric_df = pd.DataFrame(columns)
    print(metric_df)

    return metric_df

breakpoint()
qa_dataset = EmbeddingQAFinetuneDataset.from_json("/Users/julienwuthrich/GitHub/heysam_core/notebooks/pg_eval_dataset.json")
sample_id, sample_query = list(qa_dataset.queries.items())[0]
sample_expected = qa_dataset.relevant_docs[sample_id]

eval_result = retriever_evaluator.evaluate(sample_query, sample_expected)
res = []
i = 0
for query_id, query in tqdm(qa_dataset.queries.items()):
    if i == 100:
        break
    i += 1
    sample_expected = qa_dataset.relevant_docs[sample_id]
    expected_ids = qa_dataset.relevant_docs[query_id]
    eval_result = retriever_evaluator.evaluate(query, expected_ids)
    res.append(eval_result)
breakpoint()

display_results("top-2 eval", res)
