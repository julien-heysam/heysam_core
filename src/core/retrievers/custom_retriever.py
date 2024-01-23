from llama_index import QueryBundle
from llama_index.schema import NodeWithScore
from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from src import PROJECT_PATHS


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        assert mode in ("AND", "OR"), ValueError("Invalid mode.")
        self._mode = mode
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]

        return retrieve_nodes


if __name__ == '__main__':
    from llama_index import (
        VectorStoreIndex,
        SimpleKeywordTableIndex,
        SimpleDirectoryReader,
        ServiceContext,
        StorageContext,
    )
    from llama_index import get_response_synthesizer
    from llama_index.query_engine import RetrieverQueryEngine

    # load documents
    documents = SimpleDirectoryReader(PROJECT_PATHS.RAW_DATA).load_data()
    # initialize service context (set chunk size)
    service_context = ServiceContext.from_defaults(chunk_size=1024)
    node_parser = service_context.node_parser
    nodes = node_parser.get_nodes_from_documents(documents)
    # initialize storage context (by default it's in-memory)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

    # define custom retriever
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)
    keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
    custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

    # define response synthesizer
    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    custom_query_engine = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,
    )

    # vector query engine
    vector_query_engine = RetrieverQueryEngine(
        retriever=vector_retriever,
        response_synthesizer=response_synthesizer,
    )
    # keyword query engine
    keyword_query_engine = RetrieverQueryEngine(
        retriever=keyword_retriever,
        response_synthesizer=response_synthesizer,
    )

    response = custom_query_engine.retrieve(
        "What did the author do during his time at Yale?"
    )
    print(str(response))
    len(response)
    response = vector_query_engine.retrieve(
        "What did the author do during his time at Yale?"
    )
    print(str(response))
    len(response)
    response = keyword_query_engine.retrieve(
        "What did the author do during his time at Yale?"
    )
    print(str(response))
    len(response)
