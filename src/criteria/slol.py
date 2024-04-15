import torch
from torch.nn import CosineSimilarity, MSELoss



class SLOLLoss(torch.nn.Module):
    """ Soft Latent Orthogonality Loss
    """
    def __init__(self):
        super(SLOLLoss, self).__init__()
        self.cos = CosineSimilarity()
        self.mse = MSELoss()

    @staticmethod
    def latent_orthogonality_loss(embeddings):
        """
        Computes a loss that encourages orthogonality in the input embeddings.
        """
        # Compute the covariance matrix of the embeddings
        cov_matrix = torch.mm(embeddings.t(), embeddings) / embeddings.size(0)

        off_diag = cov_matrix - torch.diag(cov_matrix)
        loss = torch.norm(off_diag)

        return loss

    def forward(self, embeddings, embeddings_frozen, labels):
        """ Compute the soft latent orthogonality loss given latent features x and labels y
        """
        embeddings = embeddings.view(len(embeddings), -1)
        embeddings_frozen = embeddings_frozen.view(len(embeddings), -1)

        # 1. Compute centroids for each class
        unique_labels = torch.unique(labels)
        centroids = torch.stack([embeddings[labels == i].mean(dim=0) for i in unique_labels])
        centroids_frozen = torch.stack([embeddings_frozen[labels == i].mean(dim=0) for i in unique_labels])

        loss = self.latent_orthogonality_loss(centroids)

        # 2. Compute cosine similarity between difference vectors for all pairs of centroids
        diff_centroids = centroids.unsqueeze(1) - centroids.unsqueeze(0)
        diff_centroids_frozen = centroids_frozen.unsqueeze(1) - centroids_frozen.unsqueeze(0)
        cos_sim = self.cos(diff_centroids.view(-1, diff_centroids.shape[-1]),
                           diff_centroids_frozen.view(-1, diff_centroids_frozen.shape[-1]))
        cos_sim = cos_sim.view(diff_centroids.shape[:2])

        # 3. Exclude diagonal elements (self-pairs) and sum the rest
        loss = loss + (cos_sim * (1 - torch.eye(cos_sim.size(0), device=cos_sim.device))).sum() / 2

        return loss