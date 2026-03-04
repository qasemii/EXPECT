import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import torch

def plot_eigen_spectrum(eigvals):
    plt.title('Eigenvalue spectrum')
    plt.scatter(eigvals.cpu(), [0] * eigvals.shape[0], s=5, c='blue')

    
def visualize_mismatched_clusters(emb1, emb2, clusters, eigvals, method = "pca", figsize = (15, 10), save_path = None):
    
    emb1_cpu = emb1.cpu().float().numpy()
    emb2_cpu = emb2.cpu().float().numpy()
    n_samples = emb1_cpu.shape[0]
    
    # Get indices of embeddings that are in clusters
    all_cluster_indices = torch.cat([c.cpu() for c in clusters]).unique()
    clustered_mask = all_cluster_indices.numpy()
    n_clustered = len(clustered_mask)
    
    # Filter embeddings to only those in clusters
    emb1_filtered = emb1_cpu[clustered_mask]
    emb2_filtered = emb2_cpu[clustered_mask]
    
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        emb1_2d = reducer.fit_transform(emb1_filtered)
        emb2_2d = reducer.transform(emb2_filtered)
        print(f"Applying PCA...")
    elif method == "umap":
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, n_clustered-1))
        emb1_2d = reducer.fit_transform(emb1_filtered)
        emb2_2d = reducer.transform(emb2_filtered)
        print(f"Applying UMAP...")
    elif method == "tsne":
        from sklearn.manifold import TSNE
        print(f"Applying t-SNE...")
        reducer1 = TSNE(n_components=2, random_state=42, perplexity=min(30, n_clustered-1))
        reducer2 = TSNE(n_components=2, random_state=42, perplexity=min(30, n_clustered-1))
        emb1_2d = reducer1.fit_transform(emb1_filtered)
        emb2_2d = reducer2.fit_transform(emb2_filtered)
    
    
    # Create cluster labels for filtered embeddings
    # Map original indices to new filtered indices
    old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(clustered_mask)}
    cluster_labels = np.zeros(n_clustered, dtype=np.int64)
    
    for i, cluster in enumerate(clusters):
        cluster_cpu = cluster.cpu().numpy()
        for old_idx in cluster_cpu:
            new_idx = old_to_new_idx[old_idx]
            cluster_labels[new_idx] = i
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Mismatched Cluster Visualization ({method.upper()})', fontsize=16, fontweight='bold')
    colors = plt.cm.tab10(range(10))
    cmap = ListedColormap(colors[:len(clusters)])
    
    # Plot 1: Model 1 embeddings with clusters    
    ax = axes[0]
    scatter1 = ax.scatter(emb1_2d[:, 0], emb1_2d[:, 1], c=cluster_labels, cmap=cmap, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.set_title('Model 1 Embeddings', fontweight='bold')
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Model 2 embeddings with clusters
    ax = axes[1]
    scatter2 = ax.scatter(emb2_2d[:, 0], emb2_2d[:, 1], c=cluster_labels, cmap=cmap, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.set_title('Model 2 Embeddings', fontweight='bold')
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cluster statistics
    ax = axes[2]
    ax.axis('off')
    
    legend_elements = []
    stats_text = "Cluster Statistics:\n" + "="*40 + "\n\n"
    
    for i, cluster in enumerate(clusters):
        cluster_size = len(cluster)
        eigval = eigvals[i].item()
        
        # Compute average displacement
        cluster_cpu = cluster.cpu().numpy()
        # Map to new indices
        cluster_new_indices = [old_to_new_idx[idx] for idx in cluster_cpu]
        displacements = torch.norm(
            torch.from_numpy(emb2_2d[cluster_new_indices] - emb1_2d[cluster_new_indices]), 
            dim=1
        )
        avg_displacement = displacements.mean().item()
        
        stats_text += f"Cluster {i+1}:\n"
        stats_text += f"  Size: {cluster_size}\n"
        stats_text += f"  Eigenvalue: {eigval:.6f}\n"
        stats_text += f"  Avg displacement: {avg_displacement:.4f}\n\n"
        
        legend_elements.append(
            mpatches.Patch(color=colors[i], 
                          label=f'Cluster {i+1} (n={cluster_size}, λ={eigval:.4f})')
        )
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("VISUALIZATION SUMMARY")
    print("="*70)
    print(f"Method: {method.upper()}")
    print(f"Total samples in dataset: {n_samples}")
    print(f"Plotted (clustered) samples: {n_clustered}")
    print(f"Number of clusters: {len(clusters)}")
    
    # Compute overall displacement statistics
    all_displacements = torch.norm(
        torch.from_numpy(emb2_2d - emb1_2d), dim=1
    )
    
    print(f"\nDisplacement statistics:")
    print(f"  Clustered samples - Mean: {all_displacements.mean().item():.4f}, "
          f"Std: {all_displacements.std().item():.4f}")


# def visualize(emb1, emb2, clusters):
#     # Step 1: Collect all indices that are in clusters
#     clustered_indices = []
#     cluster_labels = []
    
#     for cluster_id, cluster_indices in enumerate(clusters):
#         # Convert tensor to numpy if needed
#         if hasattr(cluster_indices, 'numpy'):
#             indices = cluster_indices.cpu().numpy()
#         else:
#             indices = cluster_indices.cpu()
        
#         clustered_indices.extend(indices)
#         cluster_labels.extend([cluster_id] * len(indices))
    
#     # Convert to numpy arrays
#     clustered_indices = np.array(clustered_indices)
#     cluster_labels = np.array(cluster_labels)
    
#     # Step 2: Extract embeddings for both sets
#     emb1_clustered = emb1[clustered_indices]
#     emb2_clustered = emb2[clustered_indices]
    
#     # Step 3: Apply UMAP to both embedding sets
#     reducer1 = umap.UMAP(n_components=2, random_state=42)
#     embedding1_2d = reducer1.fit_transform(emb1_clustered)
    
#     reducer2 = umap.UMAP(n_components=2, random_state=42)
#     embedding2_2d = reducer2.fit_transform(emb2_clustered)
    
#     # Step 4: Create side-by-side plots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
#     # Plot emb1
#     scatter1 = ax1.scatter(embedding1_2d[:, 0], embedding1_2d[:, 1], 
#                            c=cluster_labels, cmap='tab10', s=50, alpha=0.6)
#     ax1.set_title('UMAP visualization of emb1 (clustered embeddings)', fontsize=14)
#     ax1.set_xlabel('UMAP 1')
#     ax1.set_ylabel('UMAP 2')
#     plt.colorbar(scatter1, ax=ax1, label='Cluster ID')
    
#     # Plot emb2
#     scatter2 = ax2.scatter(embedding2_2d[:, 0], embedding2_2d[:, 1], 
#                            c=cluster_labels, cmap='tab10', s=50, alpha=0.6)
#     ax2.set_title('UMAP visualization of emb2 (clustered embeddings)', fontsize=14)
#     ax2.set_xlabel('UMAP 1')
#     ax2.set_ylabel('UMAP 2')
#     plt.colorbar(scatter2, ax=ax2, label='Cluster ID')
    
#     plt.tight_layout()
#     plt.show()