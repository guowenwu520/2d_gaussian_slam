import torch

class PlaneManager:
    def __init__(self, voxel_size=0.2, min_pts=30, device="cuda"):
        self.voxel_size = voxel_size
        self.min_pts = min_pts
        self.device = device
        self.planes = []  # list of dicts: {"normal", "point"}

    def add_points(self, points: torch.Tensor):
        """
        points: [N, 3] tensor of new Gaussian centers
        Only update planes from new points, old planes are preserved.
        """
        coords = torch.floor(points / self.voxel_size)
        unique_coords, inverse_indices = torch.unique(coords, dim=0, return_inverse=True)

        for i in range(unique_coords.shape[0]):
            mask = (inverse_indices == i)
            if mask.sum() < self.min_pts:
                continue
            pts = points[mask]
            center = pts.mean(dim=0, keepdim=True)
            X = pts - center
            _, _, V = torch.svd(X)
            normal = V[:, -1]
            self.planes.append({
                "normal": normal,
                "point": center.squeeze(0),
                "indices": torch.nonzero(mask).squeeze(-1)
            })

    def compute_point2plane_loss(self, points: torch.Tensor, weights=None):
        """
        Compute point-to-plane loss for current planes.
        points: [N, 3] Gaussian centers
        weights: optional [N] or [N, 1] tensor
        Returns: scalar loss
        """
        total_loss = 0.0
        count = 0

        for plane in self.planes:
            normal = plane["normal"]  # [3]
            p0 = plane["point"]       # [3]
            indices = plane["indices"]

            if indices.numel() == 0:
                continue

            pts = points[indices]
            vec = pts - p0
            dist = torch.abs((vec @ normal).view(-1))  # [M]
            if weights is not None:
                dist = dist * weights[indices].view(-1)

            total_loss += dist.mean()
            count += 1

        return total_loss / max(count, 1)

    def clear_planes(self):
        self.planes = []

    def get_plane_info(self):
        return [{"normal": p["normal"].cpu(), "point": p["point"].cpu()} for p in self.planes]
