"""Embeddings and semantic search example.

Demonstrates generating embeddings and finding similar documents.

Usage:
    python examples/03_embeddings.py
"""

from aix import embeddings, embed, cosine_similarity, find_most_similar


def main():
    """Run embeddings examples."""
    print("=" * 60)
    print("AIX Embeddings Examples")
    print("=" * 60)

    # Example 1: Generate embeddings for multiple texts
    print("\n1. Batch Embeddings:")
    print("-" * 40)
    texts = ["cat", "dog", "bird", "fish"]
    print(f"Generating embeddings for: {texts}")
    vecs = list(embeddings(texts))
    print(f"Generated {len(vecs)} vectors")
    print(f"Vector dimension: {len(vecs[0])}")

    # Example 2: Single text embedding
    print("\n2. Single Text Embedding:")
    print("-" * 40)
    vec = embed("artificial intelligence")
    print(f"Embedding dimension: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")

    # Example 3: Compute similarity
    print("\n3. Cosine Similarity:")
    print("-" * 40)
    vec1 = embed("cat")
    vec2 = embed("kitten")
    vec3 = embed("computer")

    sim_cat_kitten = cosine_similarity(vec1, vec2)
    sim_cat_computer = cosine_similarity(vec1, vec3)

    print(f"Similarity(cat, kitten): {sim_cat_kitten:.4f}")
    print(f"Similarity(cat, computer): {sim_cat_computer:.4f}")
    print(f"\n'cat' and 'kitten' are more similar: {sim_cat_kitten > sim_cat_computer}")

    # Example 4: Semantic search
    print("\n4. Semantic Search:")
    print("-" * 40)
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "Neural networks are used in deep learning",
        "JavaScript is used for web development",
        "Data science involves statistics and programming",
    ]

    query = "What is AI and machine learning?"
    print(f"Query: {query}")
    print(f"\nSearching through {len(documents)} documents...")

    results = find_most_similar(query, documents, top_k=3)

    print("\nTop 3 most similar documents:")
    for i, (doc, similarity) in enumerate(results, 1):
        print(f"\n{i}. Similarity: {similarity:.4f}")
        print(f"   {doc}")

    # Example 5: Pre-computed query vector
    print("\n5. Search with Pre-computed Query Vector:")
    print("-" * 40)
    query_vec = embed("programming languages")
    results = find_most_similar(query_vec, documents, top_k=2)

    print("Top 2 results for 'programming languages':")
    for i, (doc, similarity) in enumerate(results, 1):
        print(f"\n{i}. Similarity: {similarity:.4f}")
        print(f"   {doc}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
