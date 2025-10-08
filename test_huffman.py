#!/usr/bin/env python3
"""
test_huffman.py - Script version of the Huffman encoding notebooks

This script applies binned Huffman encoding to scalar quantization codebooks,
permutes columns to minimize bit limit violations, and performs greedy rounding
to eliminate remaining violations.

Features:
- Quantization caching: Saves compression modules to avoid redundant quantization
- Grid search support: Run multiple parameter combinations in parallel
- Comprehensive logging and result saving

Usage:
    python test_huffman.py [--config CONFIG_PATH]

The script reads configuration from config/huffman/huffman.yaml by default.

Quantization Caching:
The script automatically caches quantization results to avoid redundant computation.
Cache configuration can be controlled via the config file:

quantization_cache:
  enabled: true          # Enable/disable caching (default: true)
  cache_dir: "quantization_cache"  # Cache directory (default: "quantization_cache")

Cache keys are generated based on:
- Weight file path
- Hessian file path  
- VQ configuration parameters

This ensures that quantization is only performed once per unique layer/VQ config combination.
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
import tqdm
import yaml
import argparse
import os
import time
import json
import logging
import sys
from collections import Counter
import matplotlib.pyplot as plt
import random
from typing import Dict, Any, Tuple, List, Union
import hashlib
import multiprocessing as mp
from multiprocessing import Process, Queue, Lock
import itertools
from datetime import datetime
import glob
import pickle

# Turn off scientific notation for better readability
torch.set_printoptions(sci_mode=False)

# Import project modules
from src.utils.normalizer import Normalizer
from src.quantize_compress import LinearVQ
from src.utils import utils
from src.huffman import HuffmanUser


def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logging to both file and console."""
    # Create logs directory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("huffman_encoding")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    log_file = os.path.join(logs_dir, "huffman_encoding.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_cache_key(weight_path: str, hessian_path: str, vq_config: Dict[str, Any]) -> str:
    """
    Generate a unique cache key based on weight path, hessian path, and VQ config.
    
    Args:
        weight_path: Path to the weight file
        hessian_path: Path to the hessian file  
        vq_config: VQ configuration dictionary
        
    Returns:
        Unique cache key string
    """
    # Create a hash of the relevant parameters
    key_data = {
        'weight_path': weight_path,
        'hessian_path': hessian_path,
        'vq_config': vq_config
    }
    
    # Convert to string and hash
    key_str = json.dumps(key_data, sort_keys=True)
    cache_key = hashlib.md5(key_str.encode()).hexdigest()
    
    return cache_key


def get_cache_path(cache_key: str, cache_dir: str = "quantization_cache") -> str:
    """
    Get the cache file path for a given cache key.
    
    Args:
        cache_key: Unique cache key
        cache_dir: Base cache directory
        
    Returns:
        Full path to cache file
    """
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{cache_key}.pkl")


def save_compression_module_cache(compression_module: LinearVQ, quantization_metrics: Dict[str, Any], 
                                cache_path: str, logger: logging.Logger) -> None:
    """
    Save compression module and metrics to cache.
    
    Args:
        compression_module: The LinearVQ compression module to cache
        quantization_metrics: Quantization metrics to cache
        cache_path: Path where to save the cache
        logger: Logger instance
    """
    try:
        # Create cache data
        cache_data = {
            'compression_module': compression_module,
            'quantization_metrics': quantization_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to pickle file
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Compression module cached to: {cache_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save compression module cache: {e}")


def load_compression_module_cache(cache_path: str, logger: logging.Logger) -> Tuple[LinearVQ, Dict[str, Any], bool]:
    """
    Load compression module and metrics from cache.
    
    Args:
        cache_path: Path to the cache file
        logger: Logger instance
        
    Returns:
        Tuple of (compression_module, quantization_metrics, success_flag)
    """
    try:
        if not os.path.exists(cache_path):
            return None, None, False
            
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        compression_module = cache_data['compression_module']
        quantization_metrics = cache_data['quantization_metrics']
        
        logger.info(f"Compression module loaded from cache: {cache_path}")
        logger.info(f"Cache timestamp: {cache_data.get('timestamp', 'unknown')}")
        
        return compression_module, quantization_metrics, True
        
    except Exception as e:
        logger.warning(f"Failed to load compression module cache: {e}")
        return None, None, False


def setup_device_and_seed(config: Dict[str, Any], logger: logging.Logger) -> torch.device:
    """Setup device and random seeds."""
    # Set CUDA launch blocking if specified
    if config.get("device", {}).get("cuda_launch_blocking", False):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        logger.info("CUDA_LAUNCH_BLOCKING set to 1")
    
    # Set random seeds
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Random seed set to: {seed}")
    
    # Setup device
    if torch.cuda.is_available() and config.get("device", {}).get("cuda_device") is not None:
        device = torch.device(f"cuda:{config['device']['cuda_device']}")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    return device


def validate_bin_size(weight: torch.Tensor, config: Dict[str, Any], logger: logging.Logger) -> bool:
    """
    Validate that bin size is appropriate for the layer dimensions.
    
    Args:
        weight: The weight tensor to validate against
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        True if bin size is valid, False otherwise
    """
    huffman_config = config["huffman"]
    bin_size = huffman_config["bin_size"]
    
    # Get layer dimensions
    layer_rows, layer_cols = weight.shape
    
    # Check if bin size is too large
    max_bin_size = layer_cols // 2
    if bin_size > max_bin_size:
        logger.error(f"Bin size too large: was {bin_size}, must be less than {max_bin_size} (layer_columns//2)")
        logger.error(f"Layer dimensions: {layer_rows} x {layer_cols}")
        return False
    
    logger.info(f"Bin size validation passed: {bin_size} <= {max_bin_size}")
    logger.info(f"Layer dimensions: {layer_rows} x {layer_cols}")
    return True


def load_model_data(config: Dict[str, Any], device: torch.device, logger: logging.Logger) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load weight and hessian data."""
    model_config = config["model"]
    
    # Load weight
    weight_path = model_config["weight_path"]
    logger.info(f"Loading weight from: {weight_path}")
    
    # Check if it's a directory (should not happen in normal operation, but for safety)
    if os.path.isdir(weight_path):
        raise ValueError(f"Weight path is a directory, not a file: {weight_path}")
    
    weight = torch.load(weight_path, map_location=device)["weight"].to(torch.float32).clone().detach()
    
    # Load hessian diagonal
    hessian_path = model_config["hessian_path"]
    logger.info(f"Loading hessian from: {hessian_path}")
    
    # Check if it's a directory (should not happen in normal operation, but for safety)
    if os.path.isdir(hessian_path):
        raise ValueError(f"Hessian path is a directory, not a file: {hessian_path}")
    
    hessian_diag = torch.load(hessian_path, map_location=device)["hessianDiag"].to(torch.float32)
    
    # Normalize hessian diagonal
    hessian_diag = hessian_diag / torch.median(hessian_diag)
    
    logger.info(f"Weight shape: {weight.shape}")
    logger.info(f"Hessian diagonal shape: {hessian_diag.shape}")
    
    return weight, hessian_diag


def perform_quantization(weight: torch.Tensor, hessian_diag: torch.Tensor, config: Dict[str, Any], logger: logging.Logger) -> Tuple[LinearVQ, Dict[str, Any]]:
    """Perform scalar quantization with caching support."""
    # Load VQ config
    with open("config/compress/vq.yaml", "r") as f:
        vq_config = yaml.safe_load(f)
    
    # Verify scalar quantization parameters
    if vq_config['kwargs']['d'] != 1:
        raise ValueError(f"Expected d=1 for scalar quantization, got d={vq_config['kwargs']['d']}")
    if vq_config['method'] != 'LinearVQ':
        raise ValueError(f"Expected method=LinearVQ, got method={vq_config['method']}")
    
    logger.info("Scalar quantization parameters verified")
    
    # Check if caching is enabled
    cache_enabled = config.get("quantization_cache", {}).get("enabled", True)
    cache_dir = config.get("quantization_cache", {}).get("cache_dir", "quantization_cache")
    
    if cache_enabled:
        # Generate cache key based on weight path, hessian path, and VQ config
        weight_path = config["model"]["weight_path"]
        hessian_path = config["model"]["hessian_path"]
        cache_key = generate_cache_key(weight_path, hessian_path, vq_config)
        cache_path = get_cache_path(cache_key, cache_dir)
        
        # Try to load from cache first
        compression_module, quantization_metrics, cache_success = load_compression_module_cache(cache_path, logger)
        
        if cache_success:
            logger.info("Using cached compression module")
            # Move to appropriate device
            compression_module = compression_module.to(weight.device)
            compression_module.hessianDiag = hessian_diag
            return compression_module, quantization_metrics
    
    # Cache miss or disabled - perform quantization
    logger.info("Performing quantization (cache miss or disabled)")
    
    # Initialize compression module
    compression_module = LinearVQ(weight=weight)
    compression_module.hessianDiag = hessian_diag
    compression_module.compress(**vq_config['kwargs'])
    
    # Get compression metrics
    compression_measure = (
        compression_module.get_n_bits(),
        compression_module.get_n_original_parameters(),
    )
    
    logger.info(f"Compression metrics: {compression_measure}")
    logger.info(f"Assignments shape: {compression_module.assignments.shape}")
    
    # Calculate reconstruction error
    reconstruction_error = compression_module.get_reconstruction_error(hessian_diag)
    logger.info(f"Reconstruction error: {reconstruction_error}")
    
    quantization_metrics = {
        "compression_measure": compression_measure,
        "reconstruction_error": reconstruction_error.item(),
        "vq_config": vq_config
    }
    
    # Save to cache if enabled
    if cache_enabled:
        save_compression_module_cache(compression_module, quantization_metrics, cache_path, logger)
    
    return compression_module, quantization_metrics


def perform_huffman_encoding(compression_module: LinearVQ, weight: torch.Tensor, config: Dict[str, Any], logger: logging.Logger) -> Tuple[HuffmanUser, np.ndarray, Dict[str, Any]]:
    """Perform Huffman encoding on the assignments."""
    huffman_config = config["huffman"]
    compression_config = config["compression"]
    
    # Calculate compression severity and bit limits
    compression_severity = compression_config["severity"]
    bin_size = huffman_config["bin_size"]
    
    # Get original number of bits from VQ config
    with open("config/compress/vq.yaml", "r") as f:
        vq_config = yaml.safe_load(f)
    
    bin_bit_limit = bin_size * (vq_config['kwargs']['n_bits'] - compression_severity)
    
    logger.info(f"Compression severity: {compression_severity}")
    logger.info(f"Bin size: {bin_size}")
    logger.info(f"Bin bit limit: {bin_bit_limit}")
    
    # Initialize Huffman encoder
    logger.info(f"Initializing Huffman encoder with strategy: {huffman_config['strategy']}")
    huffman_module = HuffmanUser(huffman_config["strategy"])
    
    # Encode assignments
    logger.info("Performing Huffman encoding...")
    res = huffman_module.encode(compression_module.assignments.cpu().numpy().reshape(weight.shape))
    bit_count_matrix = res['bit_count_matrix']
    
    logger.info("Huffman encoding stats:")
    logger.info(f"  Compression ratio: {res['stats']['compression_ratio']:.2f}x")
    logger.info(f"  Original size: {res['stats']['original_size_bits']} bits")
    logger.info(f"  Encoded size: {res['stats']['encoded_size_bits']} bits")
    logger.info(f"  Execution time: {res['stats']['execution_time']:.4f} seconds")
    
    return huffman_module, bit_count_matrix, {
        "compression_severity": compression_severity,
        "bin_size": bin_size,
        "bin_bit_limit": bin_bit_limit,
        "huffman_stats": res['stats']
    }


def perform_column_permutation(huffman_module: HuffmanUser, bit_count_matrix: np.ndarray, config: Dict[str, Any], logger: logging.Logger, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Perform column permutation to minimize violations."""
    huffman_config = config["huffman"]

     # Calculate original and target bits
    with open("config/compress/vq.yaml", "r") as f:
        vq_config = yaml.safe_load(f)
    original_bits = vq_config['kwargs']['n_bits']
        
    
    # Prepare huffman arguments
    huffman_args = {
        'bin_size': huffman_config['bin_size'],
        'bin_bit_limit': huffman_config['bin_size'] * (original_bits - config['compression']['severity']),  # Assuming 4-bit from VQ config
        'heuristic_metric': huffman_config['heuristic_metric'],
        'heuristic_algo': huffman_config['heuristic_algo'],
        'lsa_metric': huffman_config['lsa_metric'],
        'lsa_prep': huffman_config['lsa_prep'],
        'heuristic': huffman_config['heuristic'],
        'lsa': huffman_config['lsa'],
        'device': device
    }
    
    logger.info("Huffman permutation args:")
    for key, value in huffman_args.items():
        logger.info(f"  {key}: {value}")
    
    # Perform permutation
    logger.info("Performing column permutation...")
    permuted_bit_count_matrix, permuted_indices = huffman_module.column_permute(bit_count_matrix, **huffman_args)
    
    # Verify permutation integrity
    assert (bit_count_matrix[:, permuted_indices] == permuted_bit_count_matrix).all(), "Permutation verification failed"
    assert np.sum(bit_count_matrix) == np.sum(permuted_bit_count_matrix), "Bit count preservation failed"
    logger.info("Column permutation completed successfully")
    
    return permuted_bit_count_matrix, permuted_indices


def analyze_violations(huffman_module: HuffmanUser, permuted_bit_count_matrix: np.ndarray, config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Analyze violations after permutation."""
    huffman_config = config["huffman"]
    bin_size = huffman_config["bin_size"]
    with open("config/compress/vq.yaml", "r") as f:
        vq_config = yaml.safe_load(f)
    original_bits = vq_config['kwargs']['n_bits']
    bin_bit_limit = huffman_config['bin_size'] * (original_bits - config['compression']['severity'])
    
    logger.info("Analyzing violations...")
    violations = huffman_module.count_violations(permuted_bit_count_matrix, bin_size=bin_size, bin_bit_limit=bin_bit_limit)
    
    violation_stats = {
        "total_bits_subtracted": np.sum(violations),
        "total_bits_over": np.sum(violations * (violations > 0)),
        "min_positive_violation": np.min(violations * (violations > 0).astype(int)) if np.any(violations > 0) else 0,
        "max_violation": np.max(violations),
        "violations_shape": violations.shape,
        "bin_violations_percentage": np.sum(violations > 0) / np.prod(violations.shape),
        "average_huffman_length": np.mean(permuted_bit_count_matrix),
        "violations_matrix": violations
    }
    
    logger.info("Violation analysis:")
    logger.info(f"  Total bits subtracted by bin limit: {violation_stats['total_bits_subtracted']}")
    logger.info(f"  Total bits over: {violation_stats['total_bits_over']}")
    logger.info(f"  Min positive violation: {violation_stats['min_positive_violation']}")
    logger.info(f"  Max violation: {violation_stats['max_violation']}")
    logger.info(f"  Bin violations percentage: {violation_stats['bin_violations_percentage']}")
    logger.info(f"  Average Huffman length: {violation_stats['average_huffman_length']}")
    
    return violation_stats


def get_entropy(matrix: np.ndarray) -> float:
    """Calculate entropy of a matrix."""
    unique_vals, counts = np.unique(matrix, return_counts=True)
    probabilities = counts / np.prod(matrix.shape)
    
    # Calculate entropy: H = -Σ p(x) * log2(p(x))
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))  # Add epsilon for numerical stability
    return entropy


def return_grid_values(param_name: str, config: Dict[str, Any]) -> List[Any]:
    """
    Return grid values for a given parameter name.
    
    Args:
        param_name: The parameter name (e.g., 'compression.severity', 'huffman.bin_size')
        config: The configuration dictionary
        
    Returns:
        List of values to iterate over for grid search
    """
    grid_config = config.get("grid_search", {})
    
    if param_name == "compression.severity":
        # Generate values from 0 to vq_config.n_bits-1 (step of 0.25)
        with open("config/compress/vq.yaml", "r") as f:
            vq_config = yaml.safe_load(f)
        n_bits = vq_config['kwargs']['n_bits']
        return list(np.arange(0, n_bits-.75, 0.25))  # 0 to n_bits-1 in steps of 0.25
    
    elif param_name == "huffman.bin_size":
        # This should be handled specially in expand_grid_parameters
        # Return a placeholder that will be replaced with layer-specific values
        return ["layer_specific"]  # This will be replaced during expansion
    
    elif param_name == "huffman.heuristic_metric":
        return ["sum", "output_normed_sum"]
    
    elif param_name == "huffman.heuristic_algo":
        return ["round_robin", "min_max"]
    
    elif param_name == "huffman.lsa_metric":
        return ["bit_violation", "num_violation"]
    
    elif param_name == "huffman.lsa_prep":
        return ["none", "flip", "shuffle"]
    
    elif param_name == "huffman.heuristic":
        return [True, False]
    
    elif param_name == "huffman.lsa":
        return [True, False]
    
    elif param_name == "model.name":
        # Use the new discovery method to get model names
        try:
            pairs = discover_weight_hessian_pairs(config)
            model_names = list(set(pair["model_name"] for pair in pairs))
            return sorted(model_names)
        except Exception as e:
            print(f"Warning: Could not discover models: {e}")
            return []
    
    elif param_name == "model.layer_name":
        # Use the new discovery method to get layer names
        try:
            pairs = discover_weight_hessian_pairs(config)
            layer_names = [pair["layer_name"] for pair in pairs]
            return sorted(layer_names)
        except Exception as e:
            print(f"Warning: Could not discover layers: {e}")
            return []
    
    else:
        raise ValueError(f"Unknown parameter for grid search: {param_name}")


def discover_weight_hessian_pairs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Discover all valid weight-hessian pairs from directory paths.
    
    Args:
        config: Configuration dictionary containing model paths
        
    Returns:
        List of dictionaries containing model_name, layer_name, weight_path, and hessian_path
    """
    model_config = config["model"]
    weight_base_path = model_config["weight_path"]
    hessian_base_path = model_config["hessian_path"]
    
    pairs = []
    
    # Check if paths are directories
    if not os.path.isdir(weight_base_path):
        raise ValueError(f"Weight path is not a directory: {weight_base_path}")
    if not os.path.isdir(hessian_base_path):
        raise ValueError(f"Hessian path is not a directory: {hessian_base_path}")
    
    # Find all .pt files in weight directory recursively
    weight_files = []
    for root, dirs, files in os.walk(weight_base_path):
        for file in files:
            if file.endswith('.pt'):
                rel_path = os.path.relpath(os.path.join(root, file), weight_base_path)
                weight_files.append(rel_path)
    
    # Find all .pt files in hessian directory recursively
    hessian_files = []
    for root, dirs, files in os.walk(hessian_base_path):
        for file in files:
            if file.endswith('.pt'):
                rel_path = os.path.relpath(os.path.join(root, file), hessian_base_path)
                hessian_files.append(rel_path)
    
    # Find matching pairs (same relative path)
    weight_files_set = set(weight_files)
    hessian_files_set = set(hessian_files)
    common_files = weight_files_set.intersection(hessian_files_set)
    
    # Extract model name from the shared path
    # Find the common prefix starting from "models/"
    weight_path_parts = os.path.normpath(weight_base_path).split(os.sep)
    hessian_path_parts = os.path.normpath(hessian_base_path).split(os.sep)
    
    # Find where "models" appears in both paths
    models_idx_weight = None
    models_idx_hessian = None
    
    for i, part in enumerate(weight_path_parts):
        if part == "models":
            models_idx_weight = i
            break
    
    for i, part in enumerate(hessian_path_parts):
        if part == "models":
            models_idx_hessian = i
            break
    
    if models_idx_weight is None or models_idx_hessian is None:
        raise ValueError("Could not find 'models' directory in weight or hessian paths")
    
    # Extract the shared path from models/ onwards
    weight_from_models = os.sep.join(weight_path_parts[models_idx_weight:])
    hessian_from_models = os.sep.join(hessian_path_parts[models_idx_hessian:])
    
    # Find the longest common prefix
    weight_parts = weight_from_models.split(os.sep)
    hessian_parts = hessian_from_models.split(os.sep)
    
    common_parts = []
    for w_part, h_part in zip(weight_parts, hessian_parts):
        if w_part == h_part:
            common_parts.append(w_part)
        else:
            break
    
    if len(common_parts) < 2:  # At least "models" and one more directory
        raise ValueError("Could not find common path between weight and hessian directories")
    
    # Model name is the shared path with / replaced by _
    model_name = "_".join(common_parts[1:])  # Skip "models"
    
    # Create pairs for each matching file
    for file_path in sorted(common_files):
        # Layer name is the file path with / replaced by _ and .pt removed
        layer_name = file_path.replace(os.sep, "_").replace('.pt', '')
        
        # Full paths
        full_weight_path = os.path.join(weight_base_path, file_path)
        full_hessian_path = os.path.join(hessian_base_path, file_path)
        
        pairs.append({
            "model_name": model_name,
            "layer_name": layer_name,
            "weight_path": full_weight_path,
            "hessian_path": full_hessian_path
        })
    
    return pairs


def get_model_layers(model_name: str, config: Dict[str, Any]) -> List[str]:
    """Get available layers for a given model."""
    # This function is now deprecated in favor of discover_weight_hessian_pairs
    # Keeping it for backward compatibility
    pairs = discover_weight_hessian_pairs(config)
    layers = [pair["layer_name"] for pair in pairs]
    return sorted(layers)


def get_layer_specific_bin_sizes(weight_path: str) -> List[int]:
    """
    Get valid bin sizes for a specific layer based on its dimensions.
    
    Args:
        weight_path: Path to the weight file
        
    Returns:
        List of valid bin sizes (powers of 2 up to layer_columns//2)
    """
    try:
        # Load weight to get dimensions
        weight = torch.load(weight_path, map_location="cpu")["weight"]
        layer_cols = weight.shape[1]
        max_bin_size = layer_cols // 2
        
        # Generate powers of 2 up to max_bin_size
        valid_bin_sizes = []
        bin_size = 2
        while bin_size <= max_bin_size:
            valid_bin_sizes.append(bin_size)
            bin_size *= 2
        
        return valid_bin_sizes
        
    except Exception as e:
        print(f"Warning: Could not load weight file {weight_path}: {e}")
        # Return default bin sizes if loading fails
        return [2, 4, 8, 16, 32, 64, 128, 256]


def extract_model_and_layer_from_paths(weight_path: str, hessian_path: str) -> Tuple[str, str]:
    """
    Extract model name and layer name from weight and hessian file paths.
    
    Args:
        weight_path: Path to weight file
        hessian_path: Path to hessian file
        
    Returns:
        Tuple of (model_name, layer_name)
    """
    # Find the common prefix starting from "models/"
    weight_parts = os.path.normpath(weight_path).split(os.sep)
    hessian_parts = os.path.normpath(hessian_path).split(os.sep)
    
    # Find where "models" appears in both paths
    models_idx_weight = None
    models_idx_hessian = None
    
    for i, part in enumerate(weight_parts):
        if part == "models":
            models_idx_weight = i
            break
    
    for i, part in enumerate(hessian_parts):
        if part == "models":
            models_idx_hessian = i
            break
    
    if models_idx_weight is None or models_idx_hessian is None:
        raise ValueError("Could not find 'models' directory in weight or hessian paths")
    
    # Extract the shared path from models/ onwards
    weight_from_models = os.sep.join(weight_parts[models_idx_weight:])
    hessian_from_models = os.sep.join(hessian_parts[models_idx_hessian:])
    
    # Find the longest common prefix
    weight_parts = weight_from_models.split(os.sep)
    hessian_parts = hessian_from_models.split(os.sep)
    
    common_parts = []
    for w_part, h_part in zip(weight_parts, hessian_parts):
        if w_part == h_part:
            common_parts.append(w_part)
        else:
            break
    
    if len(common_parts) < 2:  # At least "models" and one more directory
        raise ValueError("Could not find common path between weight and hessian directories")
    
    # Model name is the shared path with / replaced by _
    model_name = "_".join(common_parts[1:])  # Skip "models"
    
    # Layer name is the remaining path after the common prefix
    # Get the relative path from the weight file
    weight_relative = os.path.relpath(weight_path, os.path.join(*common_parts))
    layer_name = weight_relative.replace(os.sep, "_").replace('.pt', '')
    
    return model_name, layer_name


def extract_grid_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract grid parameters from configuration for documentation purposes.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing parameter names and their possible values
    """
    grid_params = {}
    
    def find_grid_params(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, list):
                    grid_params[current_key] = value
                elif value == "grid":
                    grid_params[current_key] = return_grid_values(current_key, config)
                elif isinstance(value, dict):
                    find_grid_params(value, current_key)
    
    find_grid_params(config)
    return grid_params


def expand_grid_parameters(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand configuration with grid search parameters.
    
    Args:
        config: Base configuration dictionary
        
    Returns:
        List of configuration dictionaries for each combination
    """
    # Create a copy of the config to modify
    base_config = copy.deepcopy(config)
    
    # Determine weight-hessian pairs based on config
    weight_hessian_pairs = []
    
    # Check if weight_path and hessian_path are directories (for discovery)
    weight_path = base_config["model"]["weight_path"]
    hessian_path = base_config["model"]["hessian_path"]
    
    if os.path.isdir(weight_path) and os.path.isdir(hessian_path):
        # Directory-based discovery
        try:
            weight_hessian_pairs = discover_weight_hessian_pairs(base_config)
        except Exception as e:
            print(f"Warning: Could not discover weight-hessian pairs: {e}")
            print("Falling back to single file mode")
            return [base_config]
    else:
        # Single file or list of files mode
        # Handle both single paths and lists of paths
        if isinstance(weight_path, str):
            weight_paths = [weight_path]
        else:
            weight_paths = weight_path
            
        if isinstance(hessian_path, str):
            hessian_paths = [hessian_path]
        else:
            hessian_paths = hessian_path
        
        # Ensure we have matching pairs
        if len(weight_paths) != len(hessian_paths):
            raise ValueError(f"Mismatch between weight paths ({len(weight_paths)}) and hessian paths ({len(hessian_paths)})")
        
        # Create pairs from the provided paths
        for i, (w_path, h_path) in enumerate(zip(weight_paths, hessian_paths)):
            # Extract model name and layer name from paths
            model_name, layer_name = extract_model_and_layer_from_paths(w_path, h_path)
            
            weight_hessian_pairs.append({
                "model_name": model_name,
                "layer_name": layer_name,
                "weight_path": w_path,
                "hessian_path": h_path
            })
    
    # Find all parameters that are lists or "grid"
    grid_params = {}
    
    def find_grid_params(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, list):
                    grid_params[current_key] = value
                elif value == "grid":
                    grid_params[current_key] = return_grid_values(current_key, config)
                elif isinstance(value, dict):
                    find_grid_params(value, current_key)
    
    find_grid_params(base_config)
    
    # Handle special cases for model.name and model.layer_name
    # If we have weight-hessian pairs but no model/layer grid params, create model combinations
    if weight_hessian_pairs and ("model.name" in grid_params or "model.layer_name" in grid_params):
        # Create model combinations from discovered pairs
        model_combinations = []
        for pair in weight_hessian_pairs:
            model_config = copy.deepcopy(base_config)
            model_config["model"]["name"] = pair["model_name"]
            model_config["model"]["layer_name"] = pair["layer_name"]
            model_config["model"]["weight_path"] = pair["weight_path"]
            model_config["model"]["hessian_path"] = pair["hessian_path"]
            model_combinations.append(model_config)
        
        # Handle bin_size specially - generate layer-specific bin sizes
        if "huffman.bin_size" in grid_params:
            # Remove bin_size from grid_params since we'll handle it specially
            del grid_params["huffman.bin_size"]
            
            # Generate combinations with layer-specific bin sizes
            all_combinations = []
            for model_config in model_combinations:
                # Get valid bin sizes for this specific layer
                layer_bin_sizes = get_layer_specific_bin_sizes(model_config["model"]["weight_path"])
                
                # Generate combinations for this layer with its valid bin sizes
                if grid_params:
                    param_names = list(grid_params.keys())
                    param_values = list(grid_params.values())
                    
                    for combination in itertools.product(*param_values):
                        for bin_size in layer_bin_sizes:
                            combo_config = copy.deepcopy(model_config)
                            combo_config["huffman"]["bin_size"] = bin_size
                            
                            for param_name, param_value in zip(param_names, combination):
                                # Set nested parameter
                                keys = param_name.split('.')
                                current = combo_config
                                for key in keys[:-1]:
                                    current = current[key]
                                current[keys[-1]] = param_value
                            
                            all_combinations.append(combo_config)
                else:
                    # No other grid parameters, just iterate over bin sizes
                    for bin_size in layer_bin_sizes:
                        combo_config = copy.deepcopy(model_config)
                        combo_config["huffman"]["bin_size"] = bin_size
                        all_combinations.append(combo_config)
            
            return all_combinations
        
        # Remove model.name and model.layer_name from grid_params
        if "model.layer_name" in grid_params:
            del grid_params["model.layer_name"]
        if "model.name" in grid_params:
            del grid_params["model.name"]
        
        # Generate combinations for remaining parameters
        if grid_params:
            param_names = list(grid_params.keys())
            param_values = list(grid_params.values())
            
            all_combinations = []
            for combination in itertools.product(*param_values):
                for model_config in model_combinations:
                    combo_config = copy.deepcopy(model_config)
                    for param_name, param_value in zip(param_names, combination):
                        # Set nested parameter
                        keys = param_name.split('.')
                        current = combo_config
                        for key in keys[:-1]:
                            current = current[key]
                        current[keys[-1]] = param_value
                    
                    all_combinations.append(combo_config)
            
            return all_combinations
        else:
            return model_combinations
    
    # Handle case where we have weight-hessian pairs but no model/layer grid params
    elif weight_hessian_pairs:
        # Single file or list of files with grid search on other parameters
        model_combinations = []
        for pair in weight_hessian_pairs:
            model_config = copy.deepcopy(base_config)
            model_config["model"]["name"] = pair["model_name"]
            model_config["model"]["layer_name"] = pair["layer_name"]
            model_config["model"]["weight_path"] = pair["weight_path"]
            model_config["model"]["hessian_path"] = pair["hessian_path"]
            model_combinations.append(model_config)
        
        # Handle bin_size specially - generate layer-specific bin sizes
        if "huffman.bin_size" in grid_params:
            # Remove bin_size from grid_params since we'll handle it specially
            del grid_params["huffman.bin_size"]
            
            # Generate combinations with layer-specific bin sizes
            all_combinations = []
            for model_config in model_combinations:
                # Get valid bin sizes for this specific layer
                layer_bin_sizes = get_layer_specific_bin_sizes(model_config["model"]["weight_path"])
                
                # Generate combinations for this layer with its valid bin sizes
                if grid_params:
                    param_names = list(grid_params.keys())
                    param_values = list(grid_params.values())
                    
                    for combination in itertools.product(*param_values):
                        for bin_size in layer_bin_sizes:
                            combo_config = copy.deepcopy(model_config)
                            combo_config["huffman"]["bin_size"] = bin_size
                            
                            for param_name, param_value in zip(param_names, combination):
                                # Set nested parameter
                                keys = param_name.split('.')
                                current = combo_config
                                for key in keys[:-1]:
                                    current = current[key]
                                current[keys[-1]] = param_value
                            
                            all_combinations.append(combo_config)
                else:
                    # No other grid parameters, just iterate over bin sizes
                    for bin_size in layer_bin_sizes:
                        combo_config = copy.deepcopy(model_config)
                        combo_config["huffman"]["bin_size"] = bin_size
                        all_combinations.append(combo_config)
            
            return all_combinations
        
        # Generate combinations for remaining parameters
        if grid_params:
            param_names = list(grid_params.keys())
            param_values = list(grid_params.values())
            
            all_combinations = []
            for combination in itertools.product(*param_values):
                for model_config in model_combinations:
                    combo_config = copy.deepcopy(model_config)
                    for param_name, param_value in zip(param_names, combination):
                        # Set nested parameter
                        keys = param_name.split('.')
                        current = combo_config
                        for key in keys[:-1]:
                            current = current[key]
                        current[keys[-1]] = param_value
                    
                    all_combinations.append(combo_config)
            
            return all_combinations
        else:
            return model_combinations
    
    # Generate all combinations
    if not grid_params:
        return [base_config]
    
    param_names = list(grid_params.keys())
    param_values = list(grid_params.values())
    
    combinations = []
    for combination in itertools.product(*param_values):
        combo_config = copy.deepcopy(base_config)
        for param_name, param_value in zip(param_names, combination):
            # Set nested parameter
            keys = param_name.split('.')
            current = combo_config
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = param_value
        
        combinations.append(combo_config)
    
    return combinations


def filter_bin_sizes(combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter bin sizes based on layer dimensions.
    This function can be used to pre-filter combinations before running experiments,
    but it requires loading weight files to get actual dimensions.
    """
    filtered_combinations = []
    
    for combo in combinations:
        try:
            # Load weight to get dimensions
            weight_path = combo["model"]["weight_path"]
            weight = torch.load(weight_path, map_location="cpu")["weight"]
            
            # Check bin size
            bin_size = combo.get("huffman", {}).get("bin_size", 256)
            max_bin_size = weight.shape[1] // 2
            
            if bin_size <= max_bin_size:
                filtered_combinations.append(combo)
            else:
                print(f"Skipping combination: bin_size {bin_size} > {max_bin_size} for {weight_path}")
                
        except Exception as e:
            print(f"Warning: Could not validate bin size for {combo['model']['weight_path']}: {e}")
            # Include the combination anyway - validation will happen during execution
            filtered_combinations.append(combo)
    
    return filtered_combinations


def invert_permutation(permutation: np.ndarray) -> np.ndarray:
    """Invert a permutation array."""
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv


def test_permutation_sanity(weight: torch.Tensor, permuted_indices: np.ndarray, logger: logging.Logger) -> bool:
    """Test that permutation preserves matrix multiplication results."""
    logger.info("Testing permutation sanity...")
    x = np.random.rand(weight.shape[-1]).astype(np.float32)
    permuted_weight = weight[:, permuted_indices]
    inv_permuted_indices = invert_permutation(permuted_indices)
    no_op_weight = permuted_weight[:, inv_permuted_indices]

    no_op_is_same = torch.allclose(no_op_weight, weight, atol=1e-5)
    logger.info(f"No op weight is same: {'PASSED' if no_op_is_same else 'FAILED'}")
    
    no_perm_output = weight.cpu() @ torch.from_numpy(x)
    perm_output = permuted_weight.cpu() @ torch.from_numpy(x[permuted_indices])
    
    
    is_close = torch.allclose(no_perm_output, perm_output, atol=1e-5)
    logger.info(f"Permutation sanity check: {'PASSED' if is_close else 'FAILED'}")
    return is_close


def greedy_reduction(weight_matrix: torch.Tensor, assignment_matrix: torch.Tensor, 
                    bit_count_matrix: torch.Tensor, index: Tuple[int, int], 
                    hessian_diag: torch.Tensor, huffman_codes: Dict[int, str], 
                    bin_size: int, bin_bit_limit: float, codebook: torch.Tensor) -> None:
    """Perform greedy reduction to eliminate violations in a specific bin."""
    
    # Get length map for huffman codes
    lengths = torch.zeros(len(huffman_codes), dtype=int)
    for key, value in huffman_codes.items():
        lengths[key] = len(value)

    # Get assignments, bit count, weights, and hessian diag for the bin in question
    bin_assignment = assignment_matrix[index[0], index[1] * bin_size:(index[1]+1) * bin_size]
    bin_bit_count = bit_count_matrix[index[0], index[1] * bin_size:(index[1]+1) * bin_size]
    bin_weights = weight_matrix[index[0], index[1] * bin_size:(index[1]+1) * bin_size]
    bin_hessian_diag = hessian_diag[index[1] * bin_size: (index[1]+1) * bin_size]
    inv_bin_hessian_diag = 1/(bin_hessian_diag + 1e-8)

    bin_total_bits = torch.sum(bin_bit_count)

    # Keep rounding until under bit limit
    while bin_total_bits > bin_bit_limit:
        # Distance matrix: for each element in bin, what is the distance to each centroid
        distance_matrix = torch.abs(bin_weights.unsqueeze(1) - codebook.squeeze().unsqueeze(0))
        
        # Want closer centroids to be weighed higher in the rounding metric
        inv_distance_matrix = 1/(distance_matrix + 1e-8)
        inv_distance_matrix = inv_distance_matrix.squeeze()

        # For each element in bin, get the number of bits you are saving if you round to each centroid 
        # Clamped to not ever look at longer centroids
        length_diff = torch.clamp(bin_bit_count.unsqueeze(1) - lengths.unsqueeze(0), min=0)
        
        # The metric for which element to round is their "closeness" to each centroid,
        # multiplied by the number of bits you save, multiplied by the inv hessian diag for
        # that element, similar to Wanda (but inverted)
        weight_error_matrix = inv_distance_matrix * length_diff * inv_bin_hessian_diag.unsqueeze(1)
        
        # Get 2d argmax as solution to greedy step
        greedy_step = torch.argmax(weight_error_matrix)
        greedy_element = greedy_step // len(huffman_codes)
        greedy_new_assignment = greedy_step % len(huffman_codes)

        # Update the element in question
        bin_assignment[greedy_element] = greedy_new_assignment
        bin_bit_count[greedy_element] = lengths[greedy_new_assignment]
        bin_total_bits -= length_diff[greedy_element, greedy_new_assignment]
    
    # Update the bin
    assignment_matrix[index[0], index[1] * bin_size:(index[1]+1) * bin_size] = bin_assignment
    bit_count_matrix[index[0], index[1] * bin_size:(index[1]+1) * bin_size] = bin_bit_count


def sorted_matrix_indices(matrix: np.ndarray) -> np.ndarray:
    """Get sorted indices of matrix elements by value."""
    flat_matrix = matrix.flatten()
    
    # Get flat indices of values > 0
    positive_mask = flat_matrix > 0
    positive_indices = np.argsort(flat_matrix[positive_mask])

    # Get the original flat indices where values > 0
    original_positive_flat_indices = np.flatnonzero(positive_mask)
    sorted_flat_indices = original_positive_flat_indices[positive_indices]

    # Convert back to 2D indices
    sorted_coords = np.unravel_index(sorted_flat_indices, matrix.shape)
    return np.column_stack(sorted_coords)


def perform_greedy_rounding(compression_module: LinearVQ, weight: torch.Tensor, 
                          permuted_bit_count_matrix: np.ndarray, violations: np.ndarray,
                          huffman_module: HuffmanUser, config: Dict[str, Any], logger: logging.Logger, device: torch.device) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Perform greedy rounding to eliminate violations."""
    huffman_config = config["huffman"]
    bin_size = huffman_config["bin_size"]

    bin_bit_limit = huffman_config['bin_size'] * (4 - config['compression']['severity'])
    
    logger.info("Starting greedy rounding process...")
    
    # Convert to torch tensors
    rounded_weights = weight.clone().detach().to(device)
    assignment_matrix = compression_module.assignments.clone().detach().to(device).reshape(weight.shape)
    permuted_bit_count_matrix_torch = torch.from_numpy(permuted_bit_count_matrix)
    hessian_diag = compression_module.hessianDiag.clone().detach().to(device)
    codebook = compression_module.codebook.clone().detach().to(device)
    
    # Store original assignment matrix for comparison
    old_assignment_matrix = assignment_matrix.clone()
    
    # Get sorted bins by violations
    sorted_bins_by_violations = sorted_matrix_indices(violations)
    
    logger.info(f"Processing {len(sorted_bins_by_violations)} bins with violations...")
    
    # Apply greedy reduction to each violating bin
    for i, indices in enumerate(sorted_bins_by_violations):
        if i % 100 == 0:
            logger.info(f"Processing bin {i}/{len(sorted_bins_by_violations)}")
        
        greedy_reduction(
            rounded_weights,
            assignment_matrix,
            permuted_bit_count_matrix_torch,
            tuple(indices),
            hessian_diag,
            huffman_module.huffman_codes,
            bin_size,
            bin_bit_limit,
            codebook
        )
    
    # Verify that assignments changed
    assignment_changed = (old_assignment_matrix != assignment_matrix).any()
    logger.info(f"Assignment matrix changed: {assignment_changed}")
    
    return assignment_matrix, {
        "assignment_changed": assignment_changed,
        "num_bins_processed": len(sorted_bins_by_violations)
    }


def calculate_final_reconstruction_error(compression_module: LinearVQ, assignment_matrix: torch.Tensor, logger: logging.Logger) -> float:
    """Calculate reconstruction error after greedy rounding."""
    logger.info("Calculating final reconstruction error...")
    # Update compression module with new assignments
    compression_module.assignments = assignment_matrix.flatten().cuda()
    
    # Calculate reconstruction error
    reconstruction_error = compression_module.get_reconstruction_error(compression_module.hessianDiag)
    
    logger.info(f"Final reconstruction error: {reconstruction_error}")
    return reconstruction_error.item()


def run_single_experiment(config: Dict[str, Any], device_id: int, status_queue: Queue, 
                         device_lock: Lock, timestamp: str) -> Dict[str, Any]:
    """
    Run a single experiment with the given configuration.
    This function is designed to be run in a separate process.
    """
    try:
        # Set device for this process
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for deterministic behavior
        seed = config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Create a unique identifier for this experiment
        experiment_id = f"{config['model']['name']}_{config['model']['layer_name'].replace('/', '_')}_{config['compression']['severity']}_{config['huffman']['bin_size']}"
        
        # Create output directory at the beginning using grid search timestamp
        huffman_args = config["huffman"]
        output_dir = create_output_directory(config, huffman_args, None, timestamp)
        
        # Update status with output directory info immediately
        status_queue.put({
            'experiment_id': experiment_id,
            'status': 'started',
            'device_id': device_id,
            'timestamp': datetime.now().isoformat(),
            'output_dir': output_dir
        })
        
        # Setup logging for this process using the final output directory
        log_level = config.get("logging", {}).get("level", "INFO")
        logger = setup_logging(output_dir, log_level)
        
        logger.info(f"Starting experiment {experiment_id} on device {device_id}")
        logger.info(f"Random seed set to: {seed}")
        logger.info(f"Output directory: {output_dir}")
        
        # Run the main pipeline
        result = run_huffman_pipeline(config, device, logger, output_dir)
        
        # Update status with completion
        status_queue.put({
            'experiment_id': experiment_id,
            'status': 'completed',
            'device_id': device_id,
            'timestamp': datetime.now().isoformat(),
            'result': result,
            'output_dir': output_dir
        })
        
        return result
        
    except Exception as e:
        # Update status with error
        status_queue.put({
            'experiment_id': experiment_id if 'experiment_id' in locals() else 'unknown',
            'status': 'error',
            'device_id': device_id,
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        })
        
        return {'error': str(e)}


def run_huffman_pipeline(config: Dict[str, Any], device: torch.device, logger: logging.Logger, output_dir: str) -> Dict[str, Any]:
    """
    Run the main Huffman encoding pipeline for a single configuration.
    This is extracted from the main function to be used in parallel execution.
    """
    # Track timing
    timing = {}
    start_time = time.time()
    
    # Load model data
    weight, hessian_diag = load_model_data(config, device, logger)
    timing["data_loading"] = time.time() - start_time
    
    # Validate bin size
    if not validate_bin_size(weight, config, logger):
        logger.error("Bin size validation failed. Skipping this experiment.")
        return {
            "error": "Bin size too large for layer dimensions",
            "weight_shape": weight.shape,
            "bin_size": config["huffman"]["bin_size"],
            "max_bin_size": weight.shape[1] // 2
        }
    
    # Perform quantization
    start_time = time.time()
    compression_module, quantization_metrics = perform_quantization(weight, hessian_diag, config, logger)
    timing["quantization"] = time.time() - start_time
    
    # Perform Huffman encoding
    start_time = time.time()
    huffman_module, bit_count_matrix, huffman_metrics = perform_huffman_encoding(compression_module, weight, config, logger)
    timing["huffman_encoding"] = time.time() - start_time
    
    # Perform column permutation
    start_time = time.time()
    permuted_bit_count_matrix, permuted_indices = perform_column_permutation(huffman_module, bit_count_matrix, config, logger, device)
    timing["column_permutation"] = time.time() - start_time
    
    # Analyze violations
    start_time = time.time()
    violation_stats = analyze_violations(huffman_module, permuted_bit_count_matrix, config, logger)
    timing["violation_analysis"] = time.time() - start_time
    
    # Calculate entropy
    entropy = get_entropy(compression_module.assignments.cpu().numpy())
    logger.info(f"Entropy: {entropy}")
    
    # Test permutation sanity
    permutation_sanity_check = test_permutation_sanity(weight, permuted_indices, logger)
    
    # Perform greedy rounding
    start_time = time.time()
    assignment_matrix, greedy_rounding_metrics = perform_greedy_rounding(
        compression_module, weight, permuted_bit_count_matrix, 
        violation_stats["violations_matrix"], huffman_module, config, logger, device
    )
    timing["greedy_rounding"] = time.time() - start_time
    
    # Calculate final reconstruction error
    start_time = time.time()
    final_reconstruction_error = calculate_final_reconstruction_error(compression_module, assignment_matrix, logger)
    timing["final_error_calculation"] = time.time() - start_time
    
    reconstruction_error_increase = final_reconstruction_error - quantization_metrics['reconstruction_error']
    
    logger.info(f"Final reconstruction error: {final_reconstruction_error}")
    logger.info(f"Original reconstruction error: {quantization_metrics['reconstruction_error']}")
    logger.info(f"Reconstruction error increase: {reconstruction_error_increase}")
    
    # Prepare results
    results = {
        "quantization_metrics": quantization_metrics,
        "huffman_metrics": huffman_metrics,
        "violation_stats": violation_stats,
        "entropy": entropy,
        "permutation_sanity_check": permutation_sanity_check,
        "greedy_rounding_metrics": greedy_rounding_metrics,
        "final_reconstruction_error": final_reconstruction_error,
        "reconstruction_error_increase": reconstruction_error_increase,
        "permuted_bit_count_matrix": permuted_bit_count_matrix,
        "permuted_indices": permuted_indices,
        "bit_count_matrix": bit_count_matrix,
        "timing": timing,
        "output_dir": output_dir
    }
    
    # Save results
    save_results(output_dir, config, results, logger)
    
    logger.info("Pipeline completed successfully!")
    
    return results


def create_output_directory(config: Dict[str, Any], huffman_args: Dict[str, Any], logger: logging.Logger, timestamp: str = None) -> str:
    """Create output directory with proper structure."""
    model_config = config["model"]
    compression_config = config["compression"]
    huffman_config = config["huffman"]
    
    # Extract model name and layer name (already in the correct format from discovery)
    model_name = model_config["name"]
    layer_name = model_config["layer_name"]
    
    # Calculate original and target bits
    with open("config/compress/vq.yaml", "r") as f:
        vq_config = yaml.safe_load(f)
    original_bits = vq_config['kwargs']['n_bits']
    target_bits = original_bits - compression_config["severity"]
    
    # Create unique identifier for permutation arguments
    permutation_args = {
        'heuristic_metric': huffman_config['heuristic_metric'],
        'heuristic_algo': huffman_config['heuristic_algo'],
        'lsa_metric': huffman_config['lsa_metric'],
        'lsa_prep': huffman_config['lsa_prep'],
        'heuristic': huffman_config['heuristic'],
        'lsa': huffman_config['lsa']
    }
    
    # Create hash of permutation arguments for unique identifier
    args_str = json.dumps(permutation_args, sort_keys=True)
    unique_id = hashlib.md5(args_str.encode()).hexdigest()[:8]
    
    # Create directory structure with timestamp
    if timestamp:
        output_dir = os.path.join(
            config["output"]["base_dir"],
            model_name,
            layer_name,
            f"{original_bits}_bits",
            f"{target_bits}_bits",
            f"{huffman_config['bin_size']}_elements_per_bin",
            unique_id,
            timestamp
        )
    else:
        output_dir = os.path.join(
            config["output"]["base_dir"],
            model_name,
            layer_name,
            f"{original_bits}_bits",
            f"{target_bits}_bits",
            f"{huffman_config['bin_size']}_elements_per_bin",
            unique_id
        )
    
    os.makedirs(output_dir, exist_ok=True)
    if logger:
        logger.info(f"Output directory: {output_dir}")
    
    return output_dir


def save_results(output_dir: str, config: Dict[str, Any], results: Dict[str, Any], logger: logging.Logger) -> None:
    """Save all results to the output directory."""
    output_config = config["output"]
    
    logger.info("Saving results...")
    
    # Save configuration
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    # Save metrics
    if output_config.get("save_metrics", True):
        metrics = {
            "quantization_metrics": results["quantization_metrics"],
            "huffman_metrics": results["huffman_metrics"],
            "violation_stats": results["violation_stats"],
            "entropy": results["entropy"],
            "permutation_sanity_check": results["permutation_sanity_check"],
            "greedy_rounding_metrics": results["greedy_rounding_metrics"],
            "final_reconstruction_error": results["final_reconstruction_error"],
            "reconstruction_error_increase": results["reconstruction_error_increase"],
            "timing": results["timing"]
        }
        
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info("Metrics saved")
    
    # Save matrices
    if output_config.get("save_matrices", True):
        np.save(os.path.join(output_dir, "permuted_bit_count_matrix.npy"), results["permuted_bit_count_matrix"])
        np.save(os.path.join(output_dir, "permuted_indices.npy"), results["permuted_indices"])
        np.save(os.path.join(output_dir, "violations_matrix.npy"), results["violation_stats"]["violations_matrix"])
        logger.info("Matrices saved")
    
    # Save plots
    if output_config.get("save_plots", True):
        # Violations histogram
        plt.figure(figsize=(10, 6))
        plt.hist(results["violation_stats"]["violations_matrix"].flatten(), bins=25)
        plt.xlabel('Violation Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Bit Limit Violations')
        plt.savefig(os.path.join(output_dir, "violations_histogram.png"))
        plt.close()
        
        # Bit count distribution
        counter = Counter(results["bit_count_matrix"].flatten())
        plt.figure(figsize=(10, 6))
        plt.bar(counter.keys(), counter.values())
        plt.title('Huffman Code Length Distribution')
        plt.xlabel('Code Length (bits)')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bit_count_distribution.png"))
        plt.close()
        logger.info("Plots saved")
    
    # Save sanity check results
    if output_config.get("save_sanity_checks", True):
        sanity_checks = {
            "permutation_preserves_bits": results["permutation_sanity_check"],
            "assignment_matrix_changed": results["greedy_rounding_metrics"]["assignment_changed"],
            "entropy_calculation": results["entropy"]
        }
        
        with open(os.path.join(output_dir, "sanity_checks.json"), "w") as f:
            json.dump(sanity_checks, f, indent=2, default=str)
        logger.info("Sanity checks saved")
    
    logger.info(f"Results saved to: {output_dir}")


def run_grid_search(config: Dict[str, Any]) -> None:
    """
    Run grid search with parallel execution.
    """
    grid_config = config.get("grid_search", {})
    max_processes = grid_config.get("max_parallel_processes", 4)
    available_devices = config.get("device", {}).get("cuda_device", [0])
    
    # Generate timestamp for this grid search run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create grid search subdirectory
    grid_search_dir = os.path.join(config["output"]["base_dir"], f"grid_search_{timestamp}")
    os.makedirs(grid_search_dir, exist_ok=True)
    
    # Setup grid search logger
    grid_search_logger = setup_logging(grid_search_dir, "INFO")
    grid_search_logger.info(f"Starting grid search with timestamp: {timestamp}")
    grid_search_logger.info(f"Grid search directory: {grid_search_dir}")
    
    # Expand grid parameters
    grid_search_logger.info("Expanding grid parameters...")
    combinations = expand_grid_parameters(config)
    combinations = filter_bin_sizes(combinations)
    
    grid_search_logger.info(f"Generated {len(combinations)} parameter combinations")
    
    # Create status tracking file in grid search directory
    status_file = os.path.join(grid_search_dir, "status.json")
    
    # Create grid parameters file
    grid_parameters_file = os.path.join(grid_search_dir, "grid_parameters.json")
    grid_parameters = extract_grid_parameters(config)
    with open(grid_parameters_file, 'w') as f:
        json.dump(grid_parameters, f, indent=2, default=str)
    grid_search_logger.info(f"Grid parameters saved to: {grid_parameters_file}")
    
    # Initialize status tracking
    status_data = {
        'timestamp': timestamp,
        'total_combinations': len(combinations),
        'completed': 0,
        'failed': 0,
        'running': 0,
        'pending': len(combinations),
        'combinations': []
    }
    
    # Add combination details to status
    for i, combo in enumerate(combinations):
        experiment_id = f"{combo['model']['name']}_{combo['model']['layer_name'].replace('/', '_')}_{combo['compression']['severity']}_{combo['huffman']['bin_size']}"
        status_data['combinations'].append({
            'id': i,
            'experiment_id': experiment_id,
            'status': 'pending',
            'config': combo,
            'start_time': None,
            'end_time': None,
            'error': None,
            'result': None
        })
    
    # Save initial status
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2, default=str)
    
    grid_search_logger.info(f"Status tracking file: {status_file}")
    
    # Create multiprocessing components
    status_queue = Queue()
    device_queue = Queue()
    device_lock = Lock()
    
    if isinstance(available_devices, int):
        available_devices = [available_devices]
        
    # Initialize device queue
    for device_id in available_devices:
        device_queue.put(device_id)
    
    # Start processes
    processes = []
    active_experiments = {}
    
    def update_status():
        """Update status file with current progress."""
        try:
            while not status_queue.empty():
                try:
                    status_update = status_queue.get_nowait()
                    experiment_id = status_update['experiment_id']
                    
                    # Find the combination in status_data
                    found = False
                    for combo_status in status_data['combinations']:
                        if combo_status['experiment_id'] == experiment_id:
                            if status_update['status'] == 'started':
                                combo_status['status'] = 'running'
                                combo_status['start_time'] = status_update['timestamp']
                                # Save output directory information if provided
                                if 'output_dir' in status_update:
                                    combo_status['output_dir'] = status_update['output_dir']
                                status_data['running'] += 1
                                status_data['pending'] -= 1
                            elif status_update['status'] == 'completed':
                                combo_status['status'] = 'completed'
                                combo_status['end_time'] = status_update['timestamp']
                                combo_status['result'] = status_update.get('result')
                                # Save output directory if provided
                                if 'output_dir' in status_update:
                                    combo_status['output_dir'] = status_update['output_dir']
                                status_data['completed'] += 1
                                status_data['running'] -= 1
                            elif status_update['status'] == 'error':
                                combo_status['status'] = 'failed'
                                combo_status['end_time'] = status_update['timestamp']
                                combo_status['error'] = status_update.get('error')
                                status_data['failed'] += 1
                                status_data['running'] -= 1
                            found = True
                            break
                    
                    if not found:
                        grid_search_logger.warning(f"Could not find experiment {experiment_id} in status data")
                    
                    # Save updated status
                    with open(status_file, 'w') as f:
                        json.dump(status_data, f, indent=2, default=str)
                        
                except Exception as e:
                    grid_search_logger.error(f"Error processing status update: {e}")
                    # Continue processing other updates
                    
        except Exception as e:
            grid_search_logger.error(f"Error in update_status: {e}")
            # Don't silently ignore errors - log them
    
    # Process combinations
    combination_index = 0
    
    while combination_index < len(combinations) or processes:
        # Start new processes if we have available devices and pending combinations
        while (len(processes) < max_processes and 
               combination_index < len(combinations) and 
               not device_queue.empty()):
            
            device_id = device_queue.get()
            combo = combinations[combination_index]
            
            # Create process
            process = Process(
                target=run_single_experiment,
                args=(combo, device_id, status_queue, device_lock, timestamp)
            )
            process.start()
            processes.append((process, device_id, combination_index))
            active_experiments[combination_index] = device_id
            
            combination_index += 1
        
        # Check for completed processes
        completed_processes = []
        for i, (process, device_id, combo_idx) in enumerate(processes):
            if not process.is_alive():
                # Process completed
                process.join()  # Clean up
                device_queue.put(device_id)  # Return device to queue
                completed_processes.append(i)
                
                # Check if this process actually completed successfully
                combo = combinations[combo_idx]
                experiment_id = f"{combo['model']['name']}_{combo['model']['layer_name'].replace('/', '_')}_{combo['compression']['severity']}_{combo['huffman']['bin_size']}"
                
                # Check if results exist for this experiment using saved output directory info
                try:
                    # Find the combination in status_data to get the saved output directory
                    combo_status = None
                    for status_combo in status_data['combinations']:
                        if status_combo['experiment_id'] == experiment_id:
                            combo_status = status_combo
                            break
                    
                    if combo_status and 'output_dir' in combo_status:
                        output_dir = combo_status['output_dir']
                        
                        # Check if the output directory exists and contains config.json (indicating completion)
                        config_path = os.path.join(output_dir, "config.json")
                        if os.path.exists(config_path):
                            # Found results - mark as completed
                            grid_search_logger.info(f"Detected completed experiment: {experiment_id}")
                            status_queue.put({
                                'experiment_id': experiment_id,
                                'status': 'completed',
                                'device_id': device_id,
                                'timestamp': datetime.now().isoformat(),
                                'result': {'detected_from_filesystem': True}
                            })
                        else:
                            # No results found - mark as failed
                            grid_search_logger.warning(f"Detected failed experiment (no config.json): {experiment_id}")
                            status_queue.put({
                                'experiment_id': experiment_id,
                                'status': 'error',
                                'device_id': device_id,
                                'timestamp': datetime.now().isoformat(),
                                'error': 'Process completed but config.json not found'
                            })
                    else:
                        # No results found - mark as failed
                        grid_search_logger.warning(f"Detected failed experiment (no output directory): {experiment_id}")
                        status_queue.put({
                            'experiment_id': experiment_id,
                            'status': 'error',
                            'device_id': device_id,
                            'timestamp': datetime.now().isoformat(),
                            'error': 'Process completed but config.json not found'
                        })
                except Exception as e:
                    grid_search_logger.error(f"Error checking results for {experiment_id}: {e}")
                    # Mark as failed due to checking error
                    status_queue.put({
                        'experiment_id': experiment_id,
                        'status': 'error',
                        'device_id': device_id,
                        'timestamp': datetime.now().isoformat(),
                        'error': f'Error checking results: {str(e)}'
                    })
                
                # Remove from active experiments
                if combo_idx in active_experiments:
                    del active_experiments[combo_idx]
        
        # Remove completed processes
        for i in reversed(completed_processes):
            processes.pop(i)
        
        # Update status
        update_status()
        
        # Print progress
        if combination_index % 10 == 0 or len(processes) == 0:
            grid_search_logger.info(f"Progress: {status_data['completed']}/{status_data['total_combinations']} completed, "
                  f"{status_data['failed']} failed, {status_data['running']} running, "
                  f"{status_data['pending']} pending")
        
        # Small delay to prevent busy waiting
        time.sleep(0.1)
    
    # Final status update - drain any remaining status messages
    grid_search_logger.info("Processing final status updates...")
    final_update_attempts = 0
    max_final_updates = 100  # Prevent infinite loop
    
    while not status_queue.empty() and final_update_attempts < max_final_updates:
        update_status()
        final_update_attempts += 1
        time.sleep(0.1)  # Small delay between attempts
    
    if final_update_attempts >= max_final_updates:
        grid_search_logger.warning(f"Reached maximum final update attempts ({max_final_updates})")
    
    # Save final status
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2, default=str)
    
    grid_search_logger.info(f"Grid search completed!")
    grid_search_logger.info(f"Total combinations: {status_data['total_combinations']}")
    grid_search_logger.info(f"Completed: {status_data['completed']}")
    grid_search_logger.info(f"Failed: {status_data['failed']}")
    grid_search_logger.info(f"Status file: {status_file}")


def test_quantization_caching():
    """Test function to verify quantization caching works correctly."""
    print("Testing quantization caching...")
    
    # Create a simple test config
    test_config = {
        "model": {
            "weight_path": "test_weight.pt",
            "hessian_path": "test_hessian.pt"
        },
        "quantization_cache": {
            "enabled": True,
            "cache_dir": "test_cache"
        }
    }
    
    # Create mock VQ config
    vq_config = {
        'method': 'LinearVQ',
        'kwargs': {
            'd': 1,
            'n_bits': 4,
            'n_inits': 1,
            'n_iters': 100
        }
    }
    
    # Test cache key generation
    cache_key = generate_cache_key(
        test_config["model"]["weight_path"],
        test_config["model"]["hessian_path"], 
        vq_config
    )
    
    print(f"Generated cache key: {cache_key}")
    
    # Test cache path generation
    cache_path = get_cache_path(cache_key, test_config["quantization_cache"]["cache_dir"])
    print(f"Cache path: {cache_path}")
    
    # Clean up test cache directory
    import shutil
    if os.path.exists("test_cache"):
        shutil.rmtree("test_cache")
    
    print("Quantization caching test completed successfully!")


def main():
    """Main function to run the Huffman encoding pipeline."""
    parser = argparse.ArgumentParser(description="Run Huffman encoding on scalar quantization codebooks")
    parser.add_argument("--config", default="config/huffman/huffman.yaml", help="Path to config file")
    parser.add_argument("--test-cache", action="store_true", help="Test quantization caching functionality")
    args = parser.parse_args()
    
    if args.test_cache:
        test_quantization_caching()
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Check if grid search is enabled
    if config.get("grid_search", {}).get("enabled", False):
        print("Grid search mode enabled")
        run_grid_search(config)
    else:
        print("Single experiment mode")
        # Run single experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory at the beginning
        huffman_args = config["huffman"]
        output_dir = create_output_directory(config, huffman_args, None, timestamp)
        
        # Setup logging using the final output directory
        log_level = config.get("logging", {}).get("level", "INFO")
        logger = setup_logging(output_dir, log_level)
        
        logger.info("Starting Huffman encoding pipeline")
        logger.info(f"Using config file: {args.config}")
        logger.info(f"Output directory: {output_dir}")
        
        # Setup device and seeds
        device = setup_device_and_seed(config, logger)
        
        # Run the pipeline
        result = run_huffman_pipeline(config, device, logger, output_dir)
        
        print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
