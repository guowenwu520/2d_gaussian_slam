import torch
class ManHattanVoxelGrid:
    def __init__(self, voxel_size: float):
        self.voxel_size = voxel_size
        self.grid = {}  # key: (i, j, k) => list of {field_name: tensor.cpu()}

    def _voxel_coord(self, xyz: torch.Tensor):
        # 映射到体素坐标（xyz 可在 cuda，也可在 cpu）
        return torch.floor(xyz / self.voxel_size).int().cpu()

    def build(self, tensor_dict: dict):
        """
        tensor_dict: 包含 'xyz' 和其他属性字段，所有字段为 shape=(N, ...)
        体素化后保存在 CPU 上
        """
        self.grid.clear()
        xyz = tensor_dict["xyz"]
        voxel_coords = self._voxel_coord(xyz)
        num_gaussians = xyz.shape[0]

        for idx in range(num_gaussians):
            key = tuple(voxel_coords[idx].tolist())
            entry = {k: v[idx].detach().cpu() for k, v in tensor_dict.items()}
            # print(f"Processing {idx}/{num_gaussians} {entry}")
            if key not in self.grid:
                self.grid[key] = []
            self.grid[key].append(entry)

    def query_voxel(self, xyz: torch.Tensor):
        """
        xyz: shape=(3,) 张量（可以是 CUDA）
        返回该点所在体素内的所有高斯球条目，每个条目为字段名->GPU张量的字典
        """
        key = tuple(self._voxel_coord(xyz.unsqueeze(0))[0].tolist())
        entries = self.grid.get(key, [])
        return [{k: v for k, v in e.items()} for e in entries]

    def query_neighbors(self, xyz: torch.Tensor, radius: int = 1):
        """
        查询周围体素中所有高斯球属性，并拷贝到 CUDA
        """
        center = self._voxel_coord(xyz.unsqueeze(0))[0]
        results = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    key = (center[0] + dx, center[1] + dy, center[2] + dz)
                    entries = self.grid.get(key, [])
                    results.extend([{k: v for k, v in e.items()} for e in entries])

        return results

    def get_voxel_tensor(self, key, field_name):
        """
        获取某个体素中的所有 field_name（如 'opacity'）张量，拼接成 (N, ...) 张量
        """
        entries = self.grid.get(key, [])
        if not entries:
            return torch.empty((0,), device='cpu')
        return torch.stack([e[field_name] for e in entries])
    
    def get_all_voxel_data(self):
        """
        返回当前所有体素的完整数据：
        输出形式：Dict[Tuple[int, int, int], List[Dict[str, torch.Tensor]]]
        """
        return self.grid
