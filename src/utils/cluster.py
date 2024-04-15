import torch
import faiss
import os
def cluster(X: torch.Tensor, k=1000, max_iter=300, redo=10, check_cache=True):
    device = torch.device("cuda:0")

    if check_cache is True and os.path.exists("./cache/clusters.pt"):
        print("Found clustering in cache")
        return torch.load("./cache/clusters.pt")

    print("Compute centroids")
    latent_space_in_basis_cpu = X.cpu().numpy()
    kmeans = faiss.Kmeans(d=latent_space_in_basis_cpu.shape[1], k=k, niter=max_iter, nredo=redo, gpu=True)
    kmeans.train(latent_space_in_basis_cpu)
    print("Clustering completes")
    t = torch.tensor(kmeans.centroids).to(device)
    torch.save(t, "./cache/clusters.pt")
    return t