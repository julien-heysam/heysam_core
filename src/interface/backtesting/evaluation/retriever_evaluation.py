import re
import os
import time
import pickle
from typing import Optional, Any, Callable

import openai
import pandas as pd
from tqdm import tqdm
import cohere

from sklearn.metrics.pairwise import cosine_similarity

from llama_index.llms import OpenAI
from llama_index.schema import BaseNode
from llama_index.indices.base import BaseIndex
from llama_index.evaluation import RetrieverEvaluator
from llama_index.embeddings.base import BaseEmbedding
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.evaluation import EmbeddingQAFinetuneDataset
from llama_index import StorageContext, load_index_from_storage
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.evaluation.retrieval.base import RetrievalEvalResult, RetrievalEvalMode
from llama_index.query_engine import RetrieverQueryEngine

from src import PROJECT_API_KEYS, PROJECT_PATHS
from src.utils.regexes import RegexPatterns

openai.api_key = PROJECT_API_KEYS.OPENAI_API_KEY


class RetrievalEvaluator:

    def __init__(self, llm_name: str, embed_model: BaseEmbedding, reranker: BaseNodePostprocessor = None) -> None:
        self.llm = OpenAI(model=llm_name)
        self.embed_model = embed_model
        self.reranker = reranker
        self.metrics = ["mrr", "hit_rate"]

    @staticmethod
    def display_results(name: str, eval_results: list[RetrievalEvalResult]):
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

    def deduplicate_questions(self, qa_dataset: EmbeddingQAFinetuneDataset, max_similarity: float = .85):
        data_shape = len(qa_dataset.queries)
        queries_text = [query for _, query in qa_dataset.queries.items()]
        queries_ids = [query_id for query_id, _ in qa_dataset.queries.items()]
        embeddings = self.embed_model.get_text_embedding_batch(queries_text, show_progress=True)
        queries_ids_to_remove = []
        matrix_similarities = cosine_similarity(embeddings)
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                if i == j:
                    continue
                similarity = matrix_similarities[i][j]
                if similarity >= max_similarity:
                    queries_ids_to_remove.append(queries_ids[i])
                    queries_ids_to_remove.append(queries_ids[j])
        queries_ids_to_remove = set(queries_ids_to_remove)
        queries = {}
        relevant_docs = {}
        node_dict = {}
        nodes = list(qa_dataset.corpus.keys())
        for i in range(data_shape):
            if re.findall(RegexPatterns.URL, queries_text[i]):
                continue

            query_id = queries_ids[i]
            node_id = nodes[i]
            if query_id not in queries_ids_to_remove:
                node_dict[node_id] = qa_dataset.corpus[node_id]
                relevant_docs[query_id] = qa_dataset.relevant_docs[query_id]
                queries[query_id] = qa_dataset.queries[query_id]
        
        return EmbeddingQAFinetuneDataset(
            queries=queries, corpus=node_dict, relevant_docs=relevant_docs
        )

    def build_index(self, nodes: list[BaseNode]):
        service_context = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model)
        vector_index = VectorStoreIndex(nodes, service_context=service_context, show_progress=True)

        return vector_index

    def save_index(self, vector_index: VectorStoreIndex, persist_dir: str):
        vector_index.storage_context.persist(persist_dir=persist_dir)

    def load_index(self, persist_dir: str) -> BaseIndex:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        vector_index = load_index_from_storage(storage_context)

        return vector_index

    def load_nodes(self, path: str) -> list[BaseNode]:
        with open(path, 'rb') as file:
            nodes = pickle.load(file)
    
        return nodes

    def save_dataset(self, qa_dataset: EmbeddingQAFinetuneDataset, path: str):
        qa_dataset.save_json(path)

    def load_dataset(self, path: str) -> EmbeddingQAFinetuneDataset:
        return EmbeddingQAFinetuneDataset.from_json(path)
    
    async def aevaluate(
        self,
        query: str,
        expected_ids: list[str],
        expected_texts: Optional[list[str]] = None,
        mode: RetrievalEvalMode = RetrievalEvalMode.TEXT,
        **kwargs: Any,
    ) -> RetrievalEvalResult:
        retrieved_ids, retrieved_texts = await self._aget_retrieved_ids_and_texts(
            query, mode
        )
        breakpoint()
        reranker = kwargs.get("reranker", None)
        if reranker:
            response = reranker(model='rerank-english-v2.0', query=query, documents=retrieved_texts, top_n=5)
            retrieved_ids = [retrieved_ids[i.index] for i in response]
            retrieved_texts = [retrieved_texts[i.index] for i in response]
        metric_dict = {}
        for metric in self.metrics:
            eval_result = metric.compute(
                query, expected_ids, retrieved_ids, expected_texts, retrieved_texts
            )
            metric_dict[metric.metric_name] = eval_result

        return RetrievalEvalResult(
            query=query,
            expected_ids=expected_ids,
            expected_texts=expected_texts,
            retrieved_ids=retrieved_ids,
            retrieved_texts=retrieved_texts,
            mode=mode,
            metric_dict=metric_dict,
        )

    def evaluate(self, vector_index: BaseIndex, qa_dataset: EmbeddingQAFinetuneDataset, reranker: Callable):
        if self.reranker:
            retriever = vector_index.as_retriever(similarity_top_k=20, node_postprocessors=[cohere_rerank])
        else:
            retriever = vector_index.as_retriever(similarity_top_k=5)
        retriever_evaluator = RetrieverEvaluator.from_metric_names(self.metrics, retriever=retriever)
        # retriever_evaluator.aevaluate = self.aevaluate
        results = []
        i = 0
        for query_id, query in tqdm(qa_dataset.queries.items(), desc="evaluating"):
            i += 1
            if i % 10 == 0:
                break
            expected_ids = qa_dataset.relevant_docs[query_id]
            eval_result = retriever_evaluator.evaluate(query, expected_ids)
            results.append(eval_result)
        metric_df = self.display_results("top-2 eval", results)

        return metric_df


if __name__ == '__main__':
    from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding, VoyageEmbedding
    from llama_index.embeddings.cohereai import CohereEmbedding
    from llama_index.postprocessor.cohere_rerank import CohereRerank

    cohere_rerank = CohereRerank(api_key=PROJECT_API_KEYS.COHERE_API_KEY, top_n=3)
    use_cohere = False
    clean_dataset = False
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=PROJECT_API_KEYS.OPENAI_API_KEY, embed_batch_size=128)
    # embed_model = VoyageEmbedding(model_name="voyage-02", voyage_api_key=PROJECT_API_KEYS.VOYAGE_API_KEY)
    # embed_model = CohereEmbedding(cohere_api_key=PROJECT_API_KEYS.COHERE_API_KEY, model_name="embed-english-v3.0", input_type="search_query", embed_batch_size=64)
    # embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-large-en-v1.5')
    # embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large", embed_batch_size=8)
    co_reranker = cohere.Client('UqmDoQEJQ4u7WzgEvrrQlFaFq84fayaFYLr1LsPR')
    model_name = embed_model.model_name.split("/")[-1]
    model_path = PROJECT_PATHS.MODEL_DATA / f"512_tokens{'_cohere_reranker' if use_cohere else '_no_reranker'}" / f"vector_index_{model_name}"
    data_path = PROJECT_PATHS.INTERIM_DATA / "512_tokens"
    evaluator = RetrievalEvaluator(llm_name="gpt-4", embed_model=embed_model, reranker=cohere_rerank if use_cohere else None)
    nodes = evaluator.load_nodes(path=data_path / "nodes.pickle")
    if clean_dataset:
        qa_dataset = evaluator.load_dataset(path=data_path / "qa_dataset.json")
        new_qa_dataset = evaluator.deduplicate_questions(qa_dataset)
        evaluator.save_dataset(qa_dataset=new_qa_dataset, path=data_path / "qa_dataset_deduplicated.json")

    if os.path.exists(model_path):
        vector_index = evaluator.load_index(persist_dir=model_path)
    else:
        model_path.mkdir(parents=True)
        vector_index = evaluator.build_index(nodes=nodes)
        evaluator.save_index(vector_index=vector_index, persist_dir=model_path)

    qa_dataset = evaluator.load_dataset(path=data_path / "qa_dataset_deduplicated.json")
    metric_df = evaluator.evaluate(vector_index=vector_index, qa_dataset=qa_dataset, reranker=co_reranker.rerank)
    metric_df.to_csv(model_path / "results.csv", index=False)
