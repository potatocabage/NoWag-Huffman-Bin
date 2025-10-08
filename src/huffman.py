import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
import heapq
import time
from dahuffman import HuffmanCodec
import random
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

class HuffmanUser:

    def __init__(self, strategy, **kwargs):
        """
        Initializes the HuffmanUser with a specific encoding strategy.
        
        Args:
            strategy: a string that maps to an instance of a Huffman encoding strategy class.
        """

        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        available_strategies = ['HuffmanStrategy', 'DaHuffmanStrategy']
        if strategy not in available_strategies:
            raise ValueError("Invalid strategy. Available strategies: " + ", ".join(available_strategies))
        
        if strategy == 'HuffmanStrategy':
            self.strategy = HuffmanStrategy()
        elif strategy == 'DaHuffmanStrategy':
            self.strategy = DaHuffmanStrategy()
        
        self.stats = {}
    
    def encode(self, assignments):
        """
        Encodes the assignments using the specified strategy.
        """
        res = self.strategy.encode(assignments)
        self.stats = res['stats']
        self.bit_count_matrix = res['bit_count_matrix']
        self.huffman_codes = res['huffman_codes']
        return res
    
    def column_permute(self, bit_count_matrix, bin_size, bin_bit_limit, heuristic_metric, heuristic_algo, lsa_metric, lsa_prep, heuristic=True, lsa=True, device=None):
        #TODO: return final permutation indices for use of decoding
        """
        Permutes the columns of the matrix based on the Huffman codes.
        
        Args:
            bit_count_matrix: The matrix to permute (numpy array).
            device: CUDA device to use for torch operations.
        
        Returns:
            A permuted version of the input matrix (numpy array) and permutation indices (numpy array).
        """
        if device is None:
            device = torch.device('cpu')

        print("bit_count_matrix shape", bit_count_matrix.shape)

        n_bins = bit_count_matrix.shape[1] // bin_size

        # pad so that columns are divisible by bin_size
        pad = bin_size - (bit_count_matrix.shape[1] % bin_size) if bit_count_matrix.shape[1] % bin_size != 0 else 0
        if pad != 0:
            assert pad > 0
            bit_count_matrix = np.pad(bit_count_matrix, ((0, 0), (0, pad)), mode='constant', constant_values=0)
            n_bins += 1
            assert bit_count_matrix.shape[1] % bin_size == 0
            assert n_bins == bit_count_matrix.shape[1] // bin_size

        print("bit_count_matrix shape after padding", bit_count_matrix.shape)

        # Convert to torch tensors for computation
        bit_count_matrix_torch = torch.from_numpy(bit_count_matrix).to(device)
        final_permuted_indices = torch.arange(bit_count_matrix.shape[1], device=device)

        permuted_bit_count_matrix_torch = bit_count_matrix_torch.clone()
        if heuristic:
            # Apply heuristic-based column permutation
            if heuristic_metric == 'sum':
                # Sort columns by sum of bit counts
                col_sums = torch.sum(bit_count_matrix_torch, axis=0)
                sorted_indices = torch.argsort(col_sums, descending=True)
                # print(sorted_indices)
                print('sorted col sums', torch.sort(col_sums, descending=True)[0].cpu().numpy())
            elif heuristic_metric == 'output_normed_sum':
                output_norms = torch.linalg.norm(bit_count_matrix_torch, axis=1)
                normalized_matrix = bit_count_matrix_torch / (output_norms.reshape(-1,1) + 1e-10)  # Avoid division by zero
                col_sums = torch.sum(normalized_matrix, axis=0)
                sorted_indices = torch.argsort(col_sums, descending=True)
            # elif heuristic_metric == 'none':
            #     sorted_indices = torch.arange(bit_count_matrix.shape[1], device=device)
            else:
                raise ValueError("Invalid heuristic metric. Choose 'sum' or 'output_normed_sum'.")
        
            if heuristic_algo == "round_robin":
                # Round Robin allocation of sorted columns into bins
                for i in range(len(sorted_indices)):
                    pi = (i % n_bins) * bin_size + (i // n_bins)
                    # print(pi)
                    permuted_bit_count_matrix_torch[:, pi] = bit_count_matrix_torch[:, sorted_indices[i]]
                    final_permuted_indices[pi] = sorted_indices[i]

            #TODO: only works for even bin size
            elif heuristic_algo == "min_max":
                assert len(sorted_indices) % 2 == 0
                for i in range(len(sorted_indices)//2):
                    pi = (i % n_bins) * bin_size + (i // n_bins)
                    permuted_bit_count_matrix_torch[:, pi] = bit_count_matrix_torch[:, sorted_indices[i]]
                    final_permuted_indices[pi] = sorted_indices[i]

                    pj = ((i % n_bins)+1) * bin_size - ((i // n_bins)+1)
                    permuted_bit_count_matrix_torch[:, pj] = bit_count_matrix_torch[:, sorted_indices[-(i+1)]]
                    final_permuted_indices[pj] = sorted_indices[-(i+1)]
                    # print(f"{pi}, {pj}")
            assert torch.all(torch.sort(final_permuted_indices)[0] == torch.arange(bit_count_matrix.shape[1], device=device))
        is_permutation_after_heuristic = torch.all(torch.sort(final_permuted_indices)[0] == torch.arange(final_permuted_indices.size, device=device))
        print('is_permutation_after_heuristic', is_permutation_after_heuristic.item())
        assert is_permutation_after_heuristic

            
        if lsa:
            # Apply LSA-based column permutation
            # Currently this is iterative
            # TODO: KEEP ITERATING UNTIL AN EARLY STOP IS TRIGGERED

            num_iterations = 10
            print('num iterations', num_iterations)
            if lsa_metric == 'bit_violation':
                violations_torch = self.count_violations_torch(permuted_bit_count_matrix_torch, bin_size, bin_bit_limit)
                prev_violations = torch.sum(torch.clamp(violations_torch, min=0)).item()
            elif lsa_metric == 'num_violation':
                violations_torch = self.count_violations_torch(permuted_bit_count_matrix_torch, bin_size, bin_bit_limit)
                prev_violations = torch.sum((violations_torch > 0).int()).item()

            os.makedirs('lsa_progress_logs', exist_ok=True)
            with open(os.path.join('lsa_progress_logs', f'4_to_3__{lsa_prep}__{lsa_metric}.txt'), 'w') as f:
                for i in range(num_iterations):
                    f.write(f'{prev_violations}, ')
                    f.flush()
                    # TODO: Need to shuffle each bin in order for LSA to do anything
                    # trying to flip every other bin first
                    if lsa_prep == 'flip':
                        for bin_i in range(n_bins):
                            if bin_i % 2:
                                permuted_bit_count_matrix_torch[:,bin_i*bin_size:(bin_i+1)*bin_size] = permuted_bit_count_matrix_torch[:,(bin_i+1)*bin_size-1:bin_i*bin_size-1:-1]
                                final_permuted_indices[bin_i*bin_size:(bin_i+1)*bin_size] = final_permuted_indices[(bin_i+1)*bin_size-1:bin_i*bin_size-1:-1]
                    elif lsa_prep == 'shuffle':
                        old_sum = torch.sum(permuted_bit_count_matrix_torch)

                        for i in range(n_bins):
                            shuffled_bin = permuted_bit_count_matrix_torch[:, i*bin_size:(i+1)*bin_size].clone().T
                            shuffled_indices = final_permuted_indices[i*bin_size:(i+1)*bin_size].clone()
                            # Create combined tensor for shuffling
                            combined = torch.cat([shuffled_bin, shuffled_indices.unsqueeze(1)], dim=1)
                            # Generate random permutation indices
                            perm_indices = torch.randperm(combined.shape[0], device=device)
                            combined = combined[perm_indices]
                            # Split back
                            permuted_bit_count_matrix_torch[:, i*bin_size:(i+1)*bin_size] = combined[:,:-1].T
                            final_permuted_indices[i*bin_size:(i+1)*bin_size] = combined[:,-1]

                        new_sum = torch.sum(permuted_bit_count_matrix_torch)
                        assert torch.allclose(old_sum, new_sum)
                    elif lsa_prep != 'none':
                        raise NotImplementedError
                            
                    for lsa_column in tqdm(range(bin_size)):
                        score_matrix = torch.zeros((n_bins,n_bins), device=device)
                        
                        # can be sped up to not recacluate all bins
                        for i in range(n_bins):
                            for j in range(i+1,n_bins):
                                # print(f"swapping block {i} with block {j}", flush=True)
                                temp_bit_count_matrix = permuted_bit_count_matrix_torch.clone()
                                temp_bit_count_matrix[:,j*bin_size+lsa_column] = permuted_bit_count_matrix_torch[:,i*bin_size+lsa_column]
                                temp_bit_count_matrix[:,i*bin_size+lsa_column] = permuted_bit_count_matrix_torch[:,j*bin_size+lsa_column]
                                # assert (temp_bit_count_matrix != permuted_bit_count_matrix_torch).any()
                                if lsa_metric == "bit_violation":
                                    # Calculate violations for both bins
                                    bin_i_sum = torch.sum(temp_bit_count_matrix[:,i*bin_size:(i+1)*bin_size], axis=1)
                                    bin_j_sum = torch.sum(temp_bit_count_matrix[:,j*bin_size:(j+1)*bin_size], axis=1)
                                    orig_bin_i_sum = torch.sum(permuted_bit_count_matrix_torch[:,i*bin_size:(i+1)*bin_size], axis=1)
                                    orig_bin_j_sum = torch.sum(permuted_bit_count_matrix_torch[:,j*bin_size:(j+1)*bin_size], axis=1)
                                    
                                    score = torch.sum(torch.clamp(bin_i_sum - bin_bit_limit, min=0) - torch.clamp(orig_bin_i_sum - bin_bit_limit, min=0) + 
                                                     torch.clamp(bin_j_sum - bin_bit_limit, min=0) - torch.clamp(orig_bin_j_sum - bin_bit_limit, min=0))
                                    
                                elif lsa_metric == "num_violation":
                                    bin_i_sum = torch.sum(temp_bit_count_matrix[:,i*bin_size:(i+1)*bin_size], axis=1)
                                    bin_j_sum = torch.sum(temp_bit_count_matrix[:,j*bin_size:(j+1)*bin_size], axis=1)
                                    orig_bin_i_sum = torch.sum(permuted_bit_count_matrix_torch[:,i*bin_size:(i+1)*bin_size], axis=1)
                                    orig_bin_j_sum = torch.sum(permuted_bit_count_matrix_torch[:,j*bin_size:(j+1)*bin_size], axis=1)
                                    
                                    score = torch.sum((bin_i_sum > bin_bit_limit).int() - (orig_bin_i_sum > bin_bit_limit).int() + 
                                                     (bin_j_sum > bin_bit_limit).int() - (orig_bin_j_sum > bin_bit_limit).int())
                                
                                score_matrix[i,j] = score
                                score_matrix[j,i] = score
                        # print(score_matrix)
                        # Convert to numpy for scipy linear_sum_assignment
                        score_matrix_np = score_matrix.cpu().numpy()
                        og_bin_indices, permuted_bin_indices = linear_sum_assignment(score_matrix_np)
                        og_indices = og_bin_indices * bin_size + lsa_column
                        permuted_indices = permuted_bin_indices * bin_size + lsa_column
                        # print('permuted indices\n', permuted_indices)
                        # print('og indices\n', og_indices)
                        temp_bit_count_matrix = permuted_bit_count_matrix_torch.clone()
                        temp_indices = final_permuted_indices.clone()
                        permuted_bit_count_matrix_torch[:,og_indices] = permuted_bit_count_matrix_torch[:,permuted_indices]
                        final_permuted_indices[og_indices] = temp_indices[permuted_indices]
                        
                    if lsa_metric == 'bit_violation':
                        violations_torch = self.count_violations_torch(permuted_bit_count_matrix_torch, bin_size, bin_bit_limit)
                        curr_violations = torch.sum(torch.clamp(violations_torch, min=0)).item()
                    elif lsa_metric == 'num_violation':
                        violations_torch = self.count_violations_torch(permuted_bit_count_matrix_torch, bin_size, bin_bit_limit)
                        curr_violations = torch.sum((violations_torch > 0).int()).item()
                    
                    if curr_violations >= prev_violations:
                        break
                    prev_violations = curr_violations
        
        is_permutation_after_lsa = torch.all(torch.sort(final_permuted_indices)[0] == torch.arange(final_permuted_indices.size, device=device))
        print('is_permutation_after_lsa', is_permutation_after_lsa.item())
        assert is_permutation_after_lsa

        # Convert back to numpy arrays for compatibility with downstream code
        permuted_bit_count_matrix = permuted_bit_count_matrix_torch.cpu().numpy()
        final_permuted_indices = final_permuted_indices.cpu().numpy()

        return permuted_bit_count_matrix, final_permuted_indices

    

    def count_violations(self, bit_count_matrix, bin_size, bin_bit_limit):
        """
        Counts the number of violations in the matrix based on the bin size and bit limit.
        
        Args:
            bit_count_matrix: The matrix to check for violations.
            bin_size: The size of each bin.
            bin_bit_limit: The maximum number of bits allowed in each bin.
        
        Returns:
            A list of violation counts for each bin.
        """
        n_bins = bit_count_matrix.shape[1] // bin_size
        bin_sums = np.sum(bit_count_matrix.reshape(-1,n_bins,bin_size), axis=2)
        return bin_sums-bin_bit_limit
    
    def count_violations_torch(self, bit_count_matrix_torch, bin_size, bin_bit_limit):
        """
        Counts the number of violations in the matrix based on the bin size and bit limit using torch.
        
        Args:
            bit_count_matrix_torch: The torch tensor matrix to check for violations.
            bin_size: The size of each bin.
            bin_bit_limit: The maximum number of bits allowed in each bin.
        
        Returns:
            A torch tensor of violation counts for each bin.
        """
        n_bins = bit_count_matrix_torch.shape[1] // bin_size
        bin_sums = torch.sum(bit_count_matrix_torch.reshape(-1,n_bins,bin_size), axis=2)
        return bin_sums - bin_bit_limit
    
    # def decode_single(self, element):
    #     return self.huffman_codes[element]
    
    # def decode(self, encdoed_matrix):

        
        

        
        
class DaHuffmanStrategy:
    """
    A class for Huffman encoding elements in matrices using an external library.
    This is a wrapper around the dahuffman library for Huffman encoding.
    """
    
    def __init__(self):
        """Initialize the Huffman encoder."""
        self.codec = None
        self.stats = {}
    
    def encode(self, matrix):
        """
        Encode the matrix using the dahuffman library.
        
        Args:
            matrix: NumPy array to encode
            
        Returns:
            A dictionary with encoded data and statistics
        """
        
        start_time = time.time()
        
        # Flatten the matrix and count frequencies
        flattened = matrix.flatten()
        frequencies = dict(zip(*np.unique(flattened, return_counts=True)))
        print("Frequencies:", frequencies)
        
        # Create a Huffman codec
        self.codec = HuffmanCodec.from_frequencies(frequencies)

        self.codec.print_code_table()
        
        # Encode the flattened matrix
        encoded_data = self.codec.encode(flattened)
        
        # Calculate compression metrics
        original_size = matrix.size * np.dtype(matrix.dtype).itemsize * 8  # Size in bits
        encoded_size = len(encoded_data) * 8  # Size in bits
        
        compression_ratio = original_size / encoded_size if encoded_size > 0 else 0

        bit_count_matrix = np.zeros(flattened.shape, dtype=np.int32)
        # Mapping for bit lengths (for faster lookup)
        bit_lengths = {symbol: tup[0] for symbol, tup in self.codec.get_code_table().items()}
        # Fill bit count matrix
        for i in range(flattened.shape[0]):
            bit_count_matrix[i] = bit_lengths[flattened[i]]
        bit_count_matrix = bit_count_matrix.reshape(matrix.shape)
        
        end_time = time.time()
        
        # Store stats
        self.stats = {
            'frequencies': frequencies,
            'compression_ratio': compression_ratio,
            'original_size_bits': original_size,
            'encoded_size_bits': int(encoded_size),
            'execution_time': end_time - start_time
        }
        
        return {
            'encoded_data': encoded_data,
            'bit_count_matrix': bit_count_matrix,
            'stats': self.stats,
            'huffman_codes': self.codec.get_code_table()
        }
        


class HuffmanStrategy:
    """
    A class for Huffman encoding elements in matrices, 
    optimized for performance with large data sets.
    """
    
    class Node:
        """Inner class representing a node in the Huffman tree."""
        __slots__ = ('freq', 'symbol', 'left', 'right', 'huff')
        
        def __init__(self, freq, symbol, left=None, right=None):
            self.freq = freq      # Frequency of the symbol
            self.symbol = symbol  # Symbol value
            self.left = left      # Left child
            self.right = right    # Right child
            self.huff = ''        # Direction (0/1)
            
        def __lt__(self, other):
            # For priority queue comparison
            return self.freq < other.freq
    
    def __init__(self):
        """Initialize the Huffman encoder."""
        self.huffman_codes = {}
        self.frequencies = {}
        self.stats = {}
        self.root = None
    
    def _count_frequencies(self, matrix):
        """Count the frequency of each unique element in the matrix."""
        # For integer matrices, use more efficient methods
        if np.issubdtype(matrix.dtype, np.integer) and matrix.min() >= 0:
            max_val = matrix.max()
            if max_val < 1_000_000:  # Only use bincount for reasonable ranges
                counts = np.bincount(matrix.ravel())
                return {i: count for i, count in enumerate(counts) if count > 0}
        
        # For other types, use unique
        unique_values, counts = np.unique(matrix, return_counts=True)
        return dict(zip(unique_values, counts))
    
    def _build_huffman_tree(self):
        """Build a Huffman tree based on the frequencies."""
        nodes = []
        heapq.heapify(nodes)
        
        # Create leaf nodes
        for symbol, freq in self.frequencies.items():
            heapq.heappush(nodes, self.Node(freq, symbol))
        
        # Handle special case of only one unique symbol
        if len(nodes) == 1:
            node = heapq.heappop(nodes)
            node.huff = '0'
            return node
        
        # Build tree by combining nodes
        while len(nodes) > 1:
            left = heapq.heappop(nodes)
            right = heapq.heappop(nodes)
            
            left.huff = '0'
            right.huff = '1'
            
            # Create new internal node
            new_node = self.Node(left.freq + right.freq, None, left, right)
            heapq.heappush(nodes, new_node)
        
        return nodes[0] if nodes else None
    
    def _generate_codes(self):
        """Generate Huffman codes using an iterative approach."""
        if not self.root:
            return {}
        
        codes = {}
        stack = [(self.root, "")]
        
        while stack:
            node, code = stack.pop()
            
            # Leaf node - assign code to symbol
            if not node.left and not node.right:
                codes[node.symbol] = code
            else:
                # Push children to stack
                if node.right:
                    stack.append((node.right, code + '1'))
                if node.left:
                    stack.append((node.left, code + '0'))
        print('code    symbol')
        for s, c in codes.items():
            print(f'{c} {s}')
        
        return codes
    
    
    def encode(self, matrix):
        """
        Memory-efficient version for large matrices that doesn't store the full encoded result.
        Instead returns the bit count matrix and codes for reconstruction.
        
        Args:
            matrix: NumPy array to encode
            
        Returns:
            A dictionary with bit count matrix and encoding information
        """

        
        start_time = time.time()

        og_shape = matrix.shape
        matrix = matrix.flatten()
        
        
        # Get frequencies
        self.frequencies = self._count_frequencies(matrix)
        
        # Build Huffman tree
        self.root = self._build_huffman_tree()
        
        # Generate codes
        self.huffman_codes = self._generate_codes()
        
        # Create bit count matrix only (more memory efficient)
        bit_count_matrix = np.zeros(matrix.shape, dtype=np.int32)
        
        # Mapping for bit lengths (for faster lookup)
        bit_lengths = {symbol: len(code) for symbol, code in self.huffman_codes.items()}
        
        # Fill bit count matrix
        for i in range(matrix.shape[0]):
            bit_count_matrix[i] = bit_lengths[matrix[i]]

        bit_count_matrix = bit_count_matrix.reshape(og_shape)
        
        # Calculate compression metrics
        original_size = matrix.size * np.dtype(matrix.dtype).itemsize * 8
        encoded_size = np.sum(bit_count_matrix)
        compression_ratio = original_size / encoded_size if encoded_size > 0 else 0
        
        end_time = time.time()
        
        # Store stats
        self.stats = {
            'frequencies': self.frequencies,
            'compression_ratio': compression_ratio,
            'original_size_bits': original_size,
            'encoded_size_bits': int(encoded_size),
            'execution_time': end_time - start_time
        }
        
        return {
            'bit_count_matrix': bit_count_matrix,
            'huffman_codes': self.huffman_codes,
            'stats': self.stats
        }
    
    def get_encoding_stats(self):
        """Return detailed statistics about the encoding process."""
        if not self.huffman_codes or not self.frequencies:
            return "No encoding has been performed yet."
        
        # Prepare data for a summary table
        data = {
            'Symbol': [],
            'Frequency': [],
            'Huffman Code': [],
            'Code Length': [],
            'Bits Used': [],
            'Percentage': []
        }
        
        total_bits = self.stats['encoded_size_bits']
        
        for symbol in sorted(self.huffman_codes.keys()):
            code = self.huffman_codes[symbol]
            freq = self.frequencies[symbol]
            bits_used = len(code) * freq
            percentage = (bits_used / total_bits * 100) if total_bits > 0 else 0
            
            data['Symbol'].append(symbol)
            data['Frequency'].append(freq)
            data['Huffman Code'].append(code)
            data['Code Length'].append(len(code))
            data['Bits Used'].append(bits_used)
            data['Percentage'].append(round(percentage, 2))
        
        import pandas as pd
        return pd.DataFrame(data).sort_values('Frequency', ascending=False)
    
    def print_summary(self):
        """Print a summary of the encoding results."""
        if not self.stats:
            print("No encoding has been performed yet.")
            return
        
        print(f"Compression Summary:")
        print(f"- Original Size: {self.stats['original_size_bits']} bits")
        print(f"- Encoded Size: {self.stats['encoded_size_bits']} bits")
        print(f"- Compression Ratio: {self.stats['compression_ratio']:.2f}x")
        print(f"- Execution Time: {self.stats['execution_time']:.4f} seconds")
        print(f"- Unique Symbols: {len(self.huffman_codes)}")
        
        # Print frequency and code length distribution
        print("\nCode length distribution:")
        length_counts = {}
        for code in self.huffman_codes.values():
            length = len(code)
            length_counts[length] = length_counts.get(length, 0) + 1
        
        for length in sorted(length_counts.keys()):
            print(f"  {length} bits: {length_counts[length]} symbols")


def example_usage():
    # Create a sample matrix
    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 2, 3],
        [3, 3, 2, 1],
        [4, 1, 3, 2]
    ])
    
    print("Original Matrix:")
    print(matrix)
    
    # Create encoder and encode matrix
    encoder = HuffmanStrategy()
    result = encoder.encode(matrix)
    
    print("\nEncoded Matrix (Huffman codes):")
    print(result['encoded_matrix'])
    
    print("\nBit Count Matrix (code lengths):")
    print(result['bit_count_matrix'])
    
    # Print summary
    encoder.print_summary()
    
    # Get detailed statistics
    stats_df = encoder.get_encoding_stats()
    print("\nDetailed Encoding Statistics:")
    print(stats_df)
    
    # Test with a larger random matrix
    print("\n\nTesting with larger matrix...")
    large_matrix = np.random.randint(0, 10, size=(50, 50))
    
    encoder = HuffmanStrategy()
    large_result = encoder.encode(large_matrix)
    
    print(f"\nLarge Matrix Shape: {large_matrix.shape}")
    print(f"Bit Count Matrix Shape: {large_result['bit_count_matrix'].shape}")
    
    # Print some sample values
    print("\nSample of original values and their bit counts:")
    for i in range(min(5, large_matrix.shape[0])):
        for j in range(min(5, large_matrix.shape[1])):
            value = large_matrix[i, j]
            code = large_result['huffman_codes'][value]
            bits = large_result['bit_count_matrix'][i, j]
            print(f"({i},{j}): Value={value}, Code={code}, Bits={bits}")
    
    # Print summary for large matrix
    encoder.print_summary()



        

