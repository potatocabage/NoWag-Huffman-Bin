from src.quantization_interface import QuantizationInterface, QuantizationLoss
import torch
import yaml
from typing import Tuple
from src.quantize_compress import LinearVQ

class NoWagVQ_adapter(QuantizationInterface):
    def __init__(self, config: dict|None = None):
        if config is None:
            config = yaml.safe_load(open("config/compress/vq.yaml", "r"))
        if config.get("method") != "LinearVQ":
            raise ValueError("NoWagVQ_adapter only supports LinearVQ config, default config is at config/compress/vq.yaml")
        super().__init__(config)

    def quantize(self, weight_path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor|None]:
        '''
        Args:
            weight_path: the path to the weight
            device: the device to quantize the weight on
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor|None]: the original weight, codebook, assignment matrix, and sparse matrix (if applicable)
        '''
        original_weight, abs_weight_path = self.__load_original_weight(weight_path)
        original_weight = original_weight.to(device)
        self.compression_module = LinearVQ(weight=original_weight)
        self.compression_module.compress(**self.config["compress"]["kwargs"])

        self.codebooks[abs_weight_path] = self.compression_module.codebook
        self.assignment_matrices[abs_weight_path] = self.compression_module.assignments
        self.original_weights[abs_weight_path] = original_weight

        return original_weight, self.compression_module.codebook, self.compression_module.assignments, None

    

    