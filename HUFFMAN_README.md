# Huffman Encoding Script

This script (`test_huffman.py`) is a script version of the Huffman encoding notebooks. It applies binned Huffman encoding to scalar quantization codebooks, permutes columns to minimize bit limit violations, and performs greedy rounding to eliminate remaining violations.

## Usage

```bash
# Run with default config (single experiment)
python test_huffman.py

# Run with custom config (single experiment)
python test_huffman.py --config path/to/your/config.yaml

# Enable grid search by setting grid_search.enabled: true in config
python test_huffman.py --config config/huffman/huffman.yaml
```

## Configuration

The script reads configuration from `config/huffman/huffman.yaml` by default. The config file includes:

- **Model configuration**: Paths to weight and hessian files
- **Compression parameters**: Severity (bits to reduce)
- **Huffman parameters**: Strategy, bin size, permutation methods
- **Device settings**: List of available CUDA devices for parallel execution
- **Logging settings**: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Grid search settings**: Enable/disable grid search, parallel processes, base paths
- **Output settings**: What to save and where

## Grid Search

The script supports comprehensive grid search functionality:

### Enabling Grid Search
Set `grid_search.enabled: true` in the config file to enable grid search mode.

### Grid Parameters
Any parameter can be set to `"grid"` to automatically expand to all valid values:

- **`compression.severity`**: Values from 0.25 to 2.0 in steps of 0.25
- **`huffman.bin_size`**: Powers of 2 up to layer dimensions/2
- **`huffman.heuristic_metric`**: ["sum", "output_normed_sum"]
- **`huffman.heuristic_algo`**: ["round_robin", "min_max"]
- **`huffman.lsa_metric`**: ["bit_violation", "num_violation"]
- **`huffman.lsa_prep`**: ["none", "flip", "shuffle"]
- **`huffman.heuristic`**: [true, false]
- **`huffman.lsa`**: [true, false]
- **`model.name`**: Automatically scans base_model_path for available models
- **`model.layer_name`**: Automatically scans each model for available layers

### Parallel Execution
- **Device Management**: Round-robin assignment of CUDA devices
- **Process Management**: Configurable maximum parallel processes (default: 4)
- **Exclusive GPU Access**: Each process gets exclusive access to a GPU
- **Resource Optimization**: Automatic process spawning and cleanup

### Progress Tracking
- **Status File**: Real-time status updates saved to `huffman_outputs/grid_search_status_{timestamp}.json`
- **Progress Monitoring**: Live updates on completed, failed, running, and pending experiments
- **Error Handling**: Failed experiments don't stop the entire grid search
- **Intermediate Results**: Results saved as experiments complete

## Logging

The script includes comprehensive logging that outputs to both console and file:

- **Console output**: Real-time progress and status messages
- **Log file**: Complete detailed log saved to `{output_dir}/logs/huffman_encoding.log`
- **Log levels**: Configurable via config file (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Timestamps**: All log entries include timestamps
- **Structured format**: Consistent formatting for easy parsing

The logger automatically switches to the final output directory once it's created, ensuring all logs are saved in the correct location.

## Output Structure

### Single Experiment
```
huffman_outputs/
├── {model_name}/
│   ├── {layer_name}/
│   │   ├── {original_bits}_bits/
│   │   │   ├── {target_bits}_bits/
│   │   │   │   ├── {bin_size}_elements_per_bin/
│   │   │   │   │   ├── {unique_id}/
│   │   │   │   │   │   ├── logs/
│   │   │   │   │   │   │   └── huffman_encoding.log
│   │   │   │   │   │   ├── config.json
│   │   │   │   │   │   ├── metrics.json
│   │   │   │   │   │   ├── sanity_checks.json
│   │   │   │   │   │   ├── permuted_bit_count_matrix.npy
│   │   │   │   │   │   ├── permuted_indices.npy
│   │   │   │   │   │   ├── violations_matrix.npy
│   │   │   │   │   │   ├── violations_histogram.png
│   │   │   │   │   │   └── bit_count_distribution.png
```

### Grid Search
```
huffman_outputs/
├── grid_search_status_{timestamp}.json  # Overall grid search status
├── {model_name}/
│   ├── {layer_name}/
│   │   ├── {original_bits}_bits/
│   │   │   ├── {target_bits}_bits/
│   │   │   │   ├── {bin_size}_elements_per_bin/
│   │   │   │   │   ├── {unique_id}/
│   │   │   │   │   │   ├── {timestamp}/  # Grid search run timestamp
│   │   │   │   │   │   │   ├── logs/
│   │   │   │   │   │   │   ├── config.json
│   │   │   │   │   │   │   ├── metrics.json
│   │   │   │   │   │   │   └── ...
```

## Key Features

1. **Scalar Quantization Verification**: Ensures `d=1` and `method=LinearVQ`
2. **Huffman Encoding**: Uses either custom or external library implementation
3. **Column Permutation**: Minimizes bit limit violations using heuristics and LSA
4. **Greedy Rounding**: Eliminates remaining violations while minimizing reconstruction error
5. **Sanity Checks**: Verifies permutation integrity and memory preservation
6. **Comprehensive Metrics**: Saves timing, violation statistics, reconstruction errors, etc.
7. **Visualization**: Generates histograms and distribution plots
8. **Comprehensive Logging**: Dual output (console + file) with configurable levels
9. **Grid Search**: Automated parameter exploration with parallel execution
10. **Progress Tracking**: Real-time status monitoring and error handling

## Requirements

- PyTorch
- NumPy
- Matplotlib
- PyYAML
- scipy (for linear sum assignment)
- dahuffman (optional, for external Huffman implementation)
- Project modules: `src.utils.normalizer`, `src.quantize_compress`, `src.huffman`

## Example Config

See `config/huffman/huffman.yaml` for a complete example configuration with all available parameters including grid search settings.
