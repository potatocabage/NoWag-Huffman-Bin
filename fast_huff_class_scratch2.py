import numpy as np
import heapq
import time
import pandas as pd
from numba import jit, njit, prange
import bitarray

class HuffmanStrategy:
    """
    A class for Huffman encoding elements in matrices, 
    optimized for performance with large data sets using JIT compilation
    and bitstream storage.
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
    
    @staticmethod
    @njit(parallel=True)
    def _create_bit_count_matrix(matrix, bit_lengths_keys, bit_lengths_values):
        """
        JIT-compiled function to create the bit count matrix.
        
        Args:
            matrix: Input matrix
            bit_lengths_keys: Array of keys from bit_lengths dict
            bit_lengths_values: Array of values from bit_lengths dict
            
        Returns:
            Bit count matrix
        """
        rows, cols = matrix.shape
        bit_count_matrix = np.zeros((rows, cols), dtype=np.int32)
        
        for i in prange(rows):
            for j in range(cols):
                value = matrix[i, j]
                # Find the index of the value in the keys array
                for k in range(len(bit_lengths_keys)):
                    if bit_lengths_keys[k] == value:
                        bit_count_matrix[i, j] = bit_lengths_values[k]
                        break
        
        return bit_count_matrix
    
    @staticmethod
    @njit
    def _create_lookup_table(symbol_keys, symbol_values, max_symbol):
        """
        JIT-compiled function to create a lookup table for faster encoding.
        
        Args:
            symbol_keys: Array of symbols
            symbol_values: Array of bit lengths
            max_symbol: Maximum symbol value
            
        Returns:
            Lookup table mapping symbols to bit lengths
        """
        # Create lookup table with -1 as default (indicating not found)
        lookup = np.full(max_symbol + 1, -1, dtype=np.int32)
        
        for i in range(len(symbol_keys)):
            if symbol_keys[i] <= max_symbol:
                lookup[symbol_keys[i]] = symbol_values[i]
                
        return lookup
    
    def _encode_to_bitstream(self, matrix):
        """
        Encode the matrix to a bitstream using the Huffman codes.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Bitstream containing the encoded matrix
        """
        # Create bitarray for efficient bit manipulation
        bits = bitarray.bitarray()
        
        # Encode each element
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                symbol = matrix[i, j]
                code = self.huffman_codes[symbol]
                # Add bits one by one (more efficient ways exist but this is more readable)
                for bit in code:
                    bits.append(bit == '1')
        
        return bits
    
    @staticmethod
    @jit
    def _calculate_total_bits(bit_count_matrix):
        """JIT-compiled function to calculate total bits."""
        return np.sum(bit_count_matrix)
    
    def encode(self, matrix):
        """
        Encode the matrix using Huffman coding, with JIT acceleration
        and bitstream storage.
        
        Args:
            matrix: NumPy array to encode
            
        Returns:
            A dictionary with encoded bitstream, bit count matrix and encoding information
        """
        start_time = time.time()
        
        # Get frequencies
        self.frequencies = self._count_frequencies(matrix)
        
        # Build Huffman tree
        self.root = self._build_huffman_tree()
        
        # Generate codes
        self.huffman_codes = self._generate_codes()
        
        # Create bit length mapping for faster lookup
        bit_lengths = {symbol: len(code) for symbol, code in self.huffman_codes.items()}
        
        # Convert dictionaries to arrays for JIT compatibility
        max_symbol = max(bit_lengths.keys()) if bit_lengths else 0
        bit_length_keys = np.array(list(bit_lengths.keys()), dtype=np.int64)
        bit_length_values = np.array(list(bit_lengths.values()), dtype=np.int32)
        
        # For small matrices or when the number of unique values is high,
        # use the direct approach
        if matrix.size < 10000 or len(bit_lengths) > 0.1 * matrix.size:
            bit_count_matrix = np.zeros(matrix.shape, dtype=np.int32)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    bit_count_matrix[i, j] = bit_lengths[matrix[i, j]]
        else:
            # For larger matrices, use JIT and lookup table for better performance
            lookup_table = self._create_lookup_table(bit_length_keys, bit_length_values, max_symbol)
            
            # Create bit count matrix using JIT if matrix is large enough
            if matrix.size > 1000:
                bit_count_matrix = self._create_bit_count_matrix(matrix, bit_length_keys, bit_length_values)
            else:
                bit_count_matrix = np.zeros(matrix.shape, dtype=np.int32)
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        bit_count_matrix[i, j] = bit_lengths[matrix[i, j]]
        
        # Encode to bitstream
        bitstream = self._encode_to_bitstream(matrix)
        
        # Calculate compression metrics
        original_size = matrix.size * np.dtype(matrix.dtype).itemsize * 8
        encoded_size = self._calculate_total_bits(bit_count_matrix)
        compression_ratio = original_size / encoded_size if encoded_size > 0 else 0
        
        end_time = time.time()
        
        # Store stats
        self.stats = {
            'frequencies': self.frequencies,
            'compression_ratio': compression_ratio,
            'original_size_bits': original_size,
            'encoded_size_bits': int(encoded_size),
            'bitstream_size_bytes': len(bitstream) // 8 + (1 if len(bitstream) % 8 else 0),
            'execution_time': end_time - start_time
        }
        
        return {
            'bitstream': bitstream,
            'bit_count_matrix': bit_count_matrix,
            'huffman_codes': self.huffman_codes,
            'stats': self.stats
        }
    
    def decode(self, bitstream, shape, huffman_codes=None):
        """
        Decode a bitstream back to the original matrix.
        
        Args:
            bitstream: The encoded bitstream
            shape: Shape of the original matrix
            huffman_codes: Huffman codes for decoding (uses self.huffman_codes if None)
            
        Returns:
            Decoded matrix
        """
        if huffman_codes is None:
            huffman_codes = self.huffman_codes
        
        # Invert huffman codes for decoding
        huffman_decode = {code: symbol for symbol, code in huffman_codes.items()}
        
        # Create result matrix
        result = np.zeros(shape, dtype=np.int64)
        
        # Prepare for decoding
        bit_index = 0
        current_code = ""
        
        # Iterate through each position in the result matrix
        for i in range(shape[0]):
            for j in range(shape[1]):
                # Read bits until a valid code is found
                while current_code not in huffman_decode:
                    if bit_index >= len(bitstream):
                        raise ValueError("Bitstream ended prematurely during decoding")
                    
                    current_code += '1' if bitstream[bit_index] else '0'
                    bit_index += 1
                
                # Found a valid code, decode it
                result[i, j] = huffman_decode[current_code]
                current_code = ""
        
        return result
    
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
        print(f"- Bitstream Size: {self.stats['bitstream_size_bytes']} bytes")
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
    """Demonstrate the usage of the optimized Huffman encoder."""
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
    
    print("\nBit Count Matrix (code lengths):")
    print(result['bit_count_matrix'])
    
    print("\nBitstream (first 32 bits):")
    bitstream = result['bitstream']
    print(bitstream[:min(32, len(bitstream))])
    
    # Print summary
    encoder.print_summary()
    
    # Get detailed statistics
    stats_df = encoder.get_encoding_stats()
    print("\nDetailed Encoding Statistics:")
    print(stats_df)
    
    # Decode the bitstream back to original matrix
    decoded_matrix = encoder.decode(bitstream, matrix.shape)
    print("\nDecoded Matrix:")
    print(decoded_matrix)
    
    # Verify the decoded matrix matches the original
    is_equal = np.array_equal(matrix, decoded_matrix)
    print(f"\nDecoded matrix matches original: {is_equal}")
    
    # Test with a larger random matrix
    print("\n\nTesting with larger matrix...")
    large_matrix = np.random.randint(0, 10, size=(50, 50))
    
    encoder = HuffmanStrategy()
    large_result = encoder.encode(large_matrix)
    
    print(f"\nLarge Matrix Shape: {large_matrix.shape}")
    print(f"Bit Count Matrix Shape: {large_result['bit_count_matrix'].shape}")
    print(f"Bitstream Length: {len(large_result['bitstream'])} bits")
    
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
    
    # Test decoding performance on large matrix
    print("\nDecoding large matrix...")
    start_time = time.time()
    decoded_large = encoder.decode(large_result['bitstream'], large_matrix.shape)
    decode_time = time.time() - start_time
    
    # Verify decode
    is_equal = np.array_equal(large_matrix, decoded_large)
    print(f"Decode time: {decode_time:.4f} seconds")
    print(f"Decoded matrix matches original: {is_equal}")

    # Benchmark with an even larger matrix
    print("\n\nBenchmarking with a very large matrix (300x300)...")
    benchmark_matrix = np.random.randint(0, 20, size=(300, 300))
    
    encoder = HuffmanStrategy()
    start_time = time.time()
    benchmark_result = encoder.encode(benchmark_matrix)
    encode_time = time.time() - start_time
    
    print(f"Encode time: {encode_time:.4f} seconds")
    print(f"Bitstream size: {benchmark_result['stats']['bitstream_size_bytes']} bytes")
    print(f"Compression ratio: {benchmark_result['stats']['compression_ratio']:.2f}x")
    
    # Test how JIT compilation improves speed on second run
    print("\nRunning second encode to demonstrate JIT improvement...")
    benchmark_matrix2 = np.random.randint(0, 20, size=(300, 300))
    start_time = time.time()
    benchmark_result2 = encoder.encode(benchmark_matrix2)
    encode_time2 = time.time() - start_time
    
    print(f"Second encode time: {encode_time2:.4f} seconds")
    print(f"Speed improvement: {encode_time/encode_time2:.2f}x faster")


if __name__ == "__main__":
    example_usage()