import torch


def compute_cosine_similarity_simple(embeddings_df1, embeddings_df2, k):
    embeddings_df1 = torch.nn.functional.normalize(embeddings_df1, p=2, dim=1)
    embeddings_df2 = torch.nn.functional.normalize(embeddings_df2, p=2, dim=1)

    # Compute the cosine similarity matrix
    # embeddings_df1: (N1, D), embeddings_df2: (N2, D)
    similarity_matrix = torch.matmul(embeddings_df1, embeddings_df2.T)
    # similarity_matrix: (N1, N2)

    # Get top-k similarities and their indices for each row in similarity_matrix
    topk_similarity, topk_indices = torch.topk(similarity_matrix, k, dim=1)

    return topk_similarity, topk_indices


def compute_cosine_similarity(
    embeddings_input: torch.Tensor, embeddings_target: torch.Tensor, top_k: int
):
    """
    Compute the top K cosine similarities between input and target embeddings.

    Parameters:
    - embeddings_input (torch.Tensor): Tensor of shape (num_input, embedding_dim)
    - embeddings_target (torch.Tensor): Tensor of shape (num_target, embedding_dim)
    - top_k (int): Number of top K similarities to return

    Returns:
    - top_k_similarity (np.ndarray): Array of shape (num_input, top_k) containing similarity scores
    - top_k_indices (np.ndarray): Array of shape (num_input, top_k) containing indices of the top K most similar embeddings
    """
    # Ensure embeddings are on the same device
    device = embeddings_input.device
    embeddings_target = embeddings_target.to(device)

    # Normalize embeddings
    input_norm = torch.norm(embeddings_input, dim=1, keepdim=True)
    target_norm = torch.norm(embeddings_target, dim=1, keepdim=True)

    # Compute cosine similarity
    similarities = torch.mm(embeddings_input, embeddings_target.T) / (
        input_norm * target_norm.T
    )

    # Remove self-similarities

    min_top_k = min(top_k, similarities.shape[1])

    # Get top K scores and indices
    top_k_scores, top_k_indices = torch.topk(
        similarities, min_top_k, dim=1, largest=True, sorted=True
    )

    # Convert to numpy arrays for easier handling
    top_k_scores = top_k_scores.cpu().numpy()
    top_k_indices = top_k_indices.cpu().numpy()

    return top_k_scores, top_k_indices, similarities
