import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
from typing import Tuple, Optional, Union, List
from src.compression_parent import CompressedLinear
import src.alignment.hessian_general_align as hessian_general_align
from src.quantize_compress import LinearVQ, LinearVQ_Halving
from src.sparse_compress import SparseLinear
from src.utils.normalizer import Normalizer
from pydantic import BaseModel


class QuantizationLoss(BaseModel):
    NoWagUnweightedReconstructionError: float
    NoWagHessianDiagReconstructionError: dict
    NoWagHessianReconstructionError: dict

class QuantizationInterface(abc.ABC):
    def __init__(self, config: dict):
        self.config = config
        self.codebooks = {}
        self.assignment_matrices = {}
        self.sparse_matrices = {}
        self.original_weights = {}

    @abc.abstractmethod
    def quantize(self, weight_path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor|None]:
        '''
        Must be implemented by the subclass

        Args:
            weight_path: the path to the weight
            device: the device to quantize the weight on
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor|None]: the original weight, codebook, assignment matrix, and sparse matrix (if applicable)
        '''
        raise NotImplementedError

    def calculate_reconstruction_error(self, weight_path: str, assignment_matrix: torch.Tensor, losses_requested: List[str]|None = None) -> QuantizationLoss:
        '''
        Calculates the reconstruction error for the weight using several methods
        '''
        loss = QuantizationLoss()
        if losses_requested is None or "NoWagUnweightedReconstructionError" in losses_requested:
            loss.NoWagUnweightedReconstructionError = self.calculate_NoWagUnweightedReconstructionError(weight_path, assignment_matrix)
        if losses_requested is None or "NoWagHessianDiagReconstructionError" in losses_requested:
            loss.NoWagHessianDiagReconstructionError = self.calculate_NoWagHessianDiagReconstructionError(weight_path, assignment_matrix)
        if losses_requested is None or "NoWagHessianReconstructionError" in losses_requested:
            loss.NoWagHessianReconstructionError = self.calculate_NoWagHessianReconstructionError(weight_path, assignment_matrix)
        return loss

    def calculate_NoWagUnweightedReconstructionError(self, weight_path: str, assignment_matrix: torch.Tensor, sparse_matrix: torch.Tensor|None = None) -> float:
        original_weight, abs_weight_path = self.__load_original_weight(weight_path)
        # assert assignment_matrix.device == original_weight.device, f"Assignment matrix device {assignment_matrix.device} does not match original weight device {original_weight.device}"
        assert assignment_matrix.device == original_weight.device

        if sparse_matrix is not None:
            reconstructed_weight = sparse_matrix + self.codebooks[abs_weight_path][assignment_matrix]
        else:   
            reconstructed_weight = self.codebooks[abs_weight_path][assignment_matrix]
        return torch.mean((original_weight - reconstructed_weight) ** 2)

    def calculate_NoWagHessianDiagReconstructionError(self, weight_path: str, assignment_matrix: torch.Tensor, sparse_matrix: torch.Tensor|None = None) -> float:
        original_weight, abs_weight_path = self.__load_original_weight(weight_path)
        # assert assignment_matrix.device == original_weight.device, f"Assignment matrix device {assignment_matrix.device} does not match original weight device {original_weight.device}"
        assert assignment_matrix.device == original_weight.device

        if "hessianDiag_path" not in self.config:
            raise ValueError("Hessian diagonal basepath not found in config")
        abs_hessian_diag_path = os.path.join(self.config["hessianDiag_path"], abs_weight_path.split("original_weights")[1])
        hessian_diag = torch.load(abs_hessian_diag_path, map_location=original_weight.device)
        
        if len(hessian_diag.shape) == 1:
            hessian_diag = hessian_diag.unsqueeze(0)
        elif len(hessian_diag.shape) != 2:
            raise ValueError("Hessian diagonal must be 1D or 2D")
        elif hessian_diag.shape[0] != 1 or hessian_diag.shape[1] != original_weight.shape[1]:
            raise ValueError("Hessian diagonal must 1xN where N is the number of columns in the original weight")
        
        if sparse_matrix is not None:
            reconstructed_weight = sparse_matrix + self.codebooks[abs_weight_path][assignment_matrix]
        else:   
            reconstructed_weight = self.codebooks[abs_weight_path][assignment_matrix]
        unormalized_error = torch.mean((original_weight - reconstructed_weight) ** 2 * hessian_diag)
        return {
            "unnormalized_error": unormalized_error,
            "mean_normalized_error": unormalized_error / torch.mean(hessian_diag),
            "median_normalized_error": unormalized_error / torch.median(hessian_diag)
        }

    def calculate_NoWagHessianReconstructionError(self, weight_path: str, assignment_matrix: torch.Tensor, sparse_matrix: torch.Tensor|None = None) -> float:
        original_weight, abs_weight_path = self.__load_original_weight(weight_path)
        # assert assignment_matrix.device == original_weight.device, f"Assignment matrix device {assignment_matrix.device} does not match original weight device {original_weight.device}"
        assert assignment_matrix.device == original_weight.device
        
        if "hessian_path" not in self.config:
            raise ValueError("Hessian basepath not found in config")
        abs_hessian_path = os.path.join(self.config["hessian_path"], abs_weight_path.split("original_weights")[1])
        hessian = torch.load(abs_hessian_path, map_location=original_weight.device)
        
        if sparse_matrix is not None:
            reconstructed_weight = sparse_matrix + self.codebooks[abs_weight_path][assignment_matrix]
        else:   
            reconstructed_weight = self.codebooks[abs_weight_path][assignment_matrix]  
        unormalized_error = hessian_general_align.loss(reconstructed_weight, original_weight, hessian)
        return {
            "unnormalized_error": unormalized_error,
            "mean_normalized_error": unormalized_error / torch.mean(hessian),
            "median_normalized_error": unormalized_error / torch.median(hessian)
        }

    def __load_original_weight(self, weight_path: str) -> Tuple[torch.Tensor, str]:
        '''
        Checks and loads the original weight from the weight dict
        Should only be run after quantization is run
        Returns the original weight and the absolute path to the original weight

        Args:
            weight_path: the path to the weight

        Returns:
            Tuple[torch.Tensor, str]: the original weight and the absolute path to the original weight  
        '''
        abs_weight_path = os.path.abspath(weight_path)
        if abs_weight_path not in self.original_weights:
            raise ValueError(f"Original weight not found for {abs_weight_path} in weight dict.\n This means eithe the path is wrong or you did not run quantization on this weight.")
        return self.original_weights[abs_weight_path], abs_weight_path
