import torch
from tqdm import tqdm
import torch.nn.functional as F


def normalize_embeddings(emb, norm="l2", dim=1):
    if norm == "l1":
        return F.normalize(emb, p=1, dim=dim)
    elif norm == "l2":
        return F.normalize(emb, p=2, dim=dim)
    else:
        raise Exception("Unknown norm value.")

def compute_normalized_laplacian(adj_matrix, return_non_normalized=False):
    adj_matrix = adj_matrix.to("cuda")

    degree = adj_matrix.sum(dim=1).to_dense()
    degree_inv_sqrt = torch.where(degree > 0, torch.pow(degree, -0.5), torch.zeros_like(degree))
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    laplacian = torch.diag(degree) - adj_matrix
    normalized_laplacian = D_inv_sqrt @ laplacian @ D_inv_sqrt

    if return_non_normalized:
        return normalized_laplacian, laplacian
    return normalized_laplacian

def laplacian_difference(L1, L2):
    return L1 - L2

def calculate_distance(emb1, emb2, metric="cosine"):
    # Ensure inputs are at least 2D
    if emb1.dim() == 1: emb1 = emb1.unsqueeze(0)
    if emb2.dim() == 1: emb2 = emb2.unsqueeze(0)

    if metric == "cosine":
        # Normalize vectors
        emb1_norm = normalize_embeddings(emb1, norm="l2")
        emb2_norm = normalize_embeddings(emb2, norm="l2")
        
        # Distance = 1 - Similarity
        return 1 - torch.mm(emb1_norm, emb2_norm.T)

    elif metric == "euclidean":
        # cdist requires float32 or float64
        return torch.cdist(emb1.float(), emb2.float(), p=2) 
        
    else:
        raise ValueError(f"Unknown distance metric: {metric}. Expected 'cosine' or 'euclidean'.")

def build_knn_graph(embeddings, k = 5, metric="cosine", return_weighted=False):
    
    # Normalize vectors
    embeddings = normalize_embeddings(embeddings, norm="l2")
    embeddings = embeddings.to("cuda")

    num_samples = embeddings.shape[0]
    coo_rows = []
    coo_cols = []
    coo_values = []
    
    # Chunked distance calculation -> for less memory
    chunk_threshold = 10000 if num_samples > 100000 else 1000
    for row_start in tqdm(range(0, num_samples, chunk_threshold)):
        emb_chunk = embeddings[row_start:row_start+chunk_threshold]

        dist_matrix = calculate_distance(emb_chunk, embeddings, metric=metric)
        dist_matrix.fill_diagonal_(torch.max(dist_matrix))  #Fill diagonal with max to avoid self selection

        values, indices = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        
        for i, (val, idx) in enumerate(zip(values, indices)):

            
            coo_rows.extend([row_start+i] * len(idx))
            coo_cols.extend(idx.cpu().detach().tolist())
            coo_values.extend(val.cpu().detach().tolist())

    # assert len(coo_rows) == len(coo_cols)
    
    # Return sparse matrix
    # NOTE: regular bool matrix get memory issues past 50k samples
    if return_weighted:
        adj_matrix = torch.sparse_coo_tensor([coo_rows, coo_cols], coo_values, [num_samples, num_samples])
    else:
        adj_matrix = torch.sparse_coo_tensor([coo_rows, coo_cols], [1] * len(coo_values), [num_samples, num_samples])

    return adj_matrix

def symmetrize_matrix(matrix, mode="mean"):    
    if mode == "mean":
        return (matrix + matrix.T) / 2
    elif mode == "max":
        # NOTE: torch.maximum does not work for sparse matrices
        matrix = matrix.to_dense()
        return torch.maximum(matrix, matrix.T).to(torch.float16).to_sparse()
    else:
        raise Exception("Unrecognised mode.")
    
    

