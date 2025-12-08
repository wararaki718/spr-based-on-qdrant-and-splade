from light_splade import SpladeEncoder
from qdrant_client import QdrantClient, models

from encode import encode_documents, encode_query2vector
from utils import show_results


def main() -> None:
    client = QdrantClient(url="http://localhost:6333")
    encoder = SpladeEncoder(model_path="bizreach-inc/light-splade-japanese-28M")

    collection_name = "sparse_splade_collection"
    docs = [
        "Qdrantは高速なベクトル検索エンジンです",
        "SPLADEはスパース表現を学習します",
        "ベクトル検索の仕組みを理解しましょう", 
        "Pythonでベクトル検索エンジンを構築します",
        "QdrantとSPLADEを組み合わせて使います",
    ]

    # https://qdrant.tech/documentation/concepts/collections/#collection-with-sparse-vectors
    client.create_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={
            "text-sparse": models.SparseVectorParams(),
        },
    )
    print(f"Collection '{collection_name}' created.")

    # https://qdrant.tech/documentation/concepts/collections/#collection-with-sparse-vectors
    points = encode_documents(encoder, docs)
    # upsert
    client.upsert(
        collection_name=collection_name,
        points=points,
    )
    print(f"Upserted {len(points)} points.\n")

    # https://qdrant.tech/documentation/concepts/search/#search-api
    query = "ベクトル検索の仕組み"
    query_sparse_vector = encode_query2vector(encoder, query)
    print(f"query: '{query}'\n")

    # get search results
    search_result: models.QueryResponse = client.query_points(
        collection_name=collection_name,
        query=query_sparse_vector,
        using="text-sparse",
    )
    show_results(search_result.points)

    # delete collection
    client.delete_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' deleted.")

    print("DONE")


if __name__ == "__main__":
    main()
