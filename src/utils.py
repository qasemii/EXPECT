
import torch

def topk_eigenpairs(matrix, k):
    n = matrix.shape[0]

    eigvals, eigvecs = torch.linalg.eigh(matrix)
    abs_eigvals = torch.abs(eigvals)
    indices = torch.argsort(abs_eigvals, descending=True)[:k]
    
    return eigvals[indices], eigvecs[:, indices]

def ky_fan_k_norm(eigvals, k):
    abs_eigvals = torch.abs(eigvals)
    
    n = len(abs_eigvals)
    if k < 1 or k > n:
        raise ValueError(f"k must be between 1 and {n}, got {k}")
    
    top_k_eigvals, _ = torch.topk(abs_eigvals, k)
    norm = torch.sum(top_k_eigvals)
    
    return norm

def xpec_discrepancy(DeltaL, k, mode = "kyfan"):
    if mode == "kyfan":
        eigvals, _ = topk_eigenpairs(DeltaL, k)
        return ky_fan_k_norm(eigvals, k)
    
    elif mode == "lse":
        eigvals, _ = topk_eigenpairs(DeltaL, k)
        abs_eigvals = torch.abs(eigvals)
        return torch.logsumexp(abs_eigvals, dim=0).item()
        
    raise ValueError(f"Unknown mode: {mode}")

def extract_mismatched_clusters(eigvecs, threshold = None, top_n = None):

    clusters = []
    
    for i in range(eigvecs.shape[1]):
        eigvec = eigvecs[:, i]
        abs_eigvec = torch.abs(eigvec)
        
        if threshold is not None:
            mask = abs_eigvec > threshold
            cluster_indices = torch.where(mask)[0]
        elif top_n is not None:
            _, cluster_indices = torch.topk(abs_eigvec, min(top_n, len(eigvec)))
        else:
            raise ValueError("Exactly one of 'threshold' or 'top_n' must be provided")
        
        if len(cluster_indices) > 0:
            clusters.append(cluster_indices)
    
    return clusters