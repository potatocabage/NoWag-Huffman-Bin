import numpy as np
import heapq
import time
import pandas as pd
from numba import jit

class HuffmanEncoder:
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
        
        return codes
    
    def encode(self, matrix):
        """
        Encode a matrix using Huffman coding.
        
        Args:
            matrix: NumPy array to encode
            
        Returns:
            A dictionary with encoded matrix, bit count matrix, and stats
        """
        start_time = time.time()
        
        # Get frequencies
        self.frequencies = self._count_frequencies(matrix)
        
        # Build Huffman tree
        self.root = self._build_huffman_tree()
        
        # Generate codes
        self.huffman_codes = self._generate_codes()
        
        # Create encoded matrix and bit count matrix
        encoded_matrix = np.empty(matrix.shape, dtype=object)
        bit_count_matrix = np.empty(matrix.shape, dtype=np.int32)
        
        # Mapping for bit lengths (for faster lookup)
        bit_lengths = {symbol: len(code) for symbol, code in self.huffman_codes.items()}
        
        # Fill matrices
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                encoded_matrix[i, j] = self.huffman_codes[value]
                bit_count_matrix[i, j] = bit_lengths[value]
        
        # Calculate compression metrics
        original_size = matrix.size * np.dtype(matrix.dtype).itemsize * 8  # Size in bits
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
            'encoded_matrix': encoded_matrix,
            'bit_count_matrix': bit_count_matrix,
            'huffman_codes': self.huffman_codes,
            'stats': self.stats
        }
    
    def encode_large(self, matrix):
        """
        Memory-efficient version for large matrices that doesn't store the full encoded result.
        Instead returns the bit count matrix and codes for reconstruction.
        
        Args:
            matrix: NumPy array to encode
            
        Returns:
            A dictionary with bit count matrix and encoding information
        """
        start_time = time.time()
        
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
            for j in range(matrix.shape[1]):
                bit_count_matrix[i, j] = bit_lengths[matrix[i, j]]
        
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
    encoder = HuffmanEncoder()
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
    
    encoder = HuffmanEncoder()
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


if __name__ == "__main__":
    example_usage()