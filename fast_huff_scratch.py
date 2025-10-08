import numpy as np
import heapq
from collections import Counter
import pandas as pd
from numba import jit, prange
import time

class Node:
    __slots__ = ('freq', 'symbol', 'left', 'right', 'huff')
    
    def __init__(self, freq, symbol, left=None, right=None):
        # Frequency of the symbol
        self.freq = freq
        
        # Symbol name (character or element value)
        self.symbol = symbol
        
        # Left node
        self.left = left
        
        # Right node
        self.right = right
        
        # Tree direction (0/1), used for encoding
        self.huff = ''
        
    def __lt__(self, nxt):
        # Define less than method for priority queue
        return self.freq < nxt.freq

def create_huffman_tree(frequency_dict):
    """Create a Huffman tree from a frequency dictionary."""
    # Pre-allocate the heap with the right size for better performance
    nodes = []
    heapq.heapify(nodes)
    
    # Convert frequency dictionary to a list of Node objects more efficiently
    for symbol, freq in frequency_dict.items():
        heapq.heappush(nodes, Node(freq, symbol))
    
    # If there's only one unique symbol, assign it a code of '0'
    if len(nodes) == 1:
        node = heapq.heappop(nodes)
        node.huff = '0'
        return node
    
    # Build the Huffman tree
    while len(nodes) > 1:
        # Extract two nodes with the lowest frequencies
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)
        
        # Assign directional codes (0 for left, 1 for right)
        left.huff = '0'
        right.huff = '1'
        
        # Create a new internal node with these two nodes as children
        # and with frequency equal to the sum of the two nodes' frequencies
        # Use a more efficient representation for internal nodes
        new_node = Node(left.freq + right.freq, None, left, right)
        
        heapq.heappush(nodes, new_node)
    
    return nodes[0]

def generate_huffman_codes(root):
    """Generate Huffman codes for each symbol based on the tree."""
    mapping = {}
    stack = [(root, "")]
    
    # Iterative approach instead of recursive for better performance
    while stack:
        node, code = stack.pop()
        
        # If leaf node, assign the current code to the symbol
        if not node.left and not node.right:
            mapping[node.symbol] = code
        else:
            if node.right:
                stack.append((node.right, code + '1'))
            if node.left:
                stack.append((node.left, code + '0'))
    
    return mapping

def count_frequencies(arr):
    """Count frequencies using numpy's unique function for better performance."""
    unique_values, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique_values, counts))

def huffman_encode_matrix(matrix):
    """Perform Huffman encoding on the elements of a matrix."""
    start_time = time.time()
    
    # Convert matrix to a 1D array for easier counting
    flattened = matrix.flatten()
    
    # Count frequency of each element using optimized counter
    frequencies = count_frequencies(flattened)
    
    # Build Huffman tree
    root = create_huffman_tree(frequencies)
    
    # Generate Huffman codes with iterative approach
    huffman_codes = generate_huffman_codes(root)
    
    # Use vectorized operations for encoding matrix
    # First create a lookup array for quick encoding
    unique_elements = list(huffman_codes.keys())
    encoded_values = [huffman_codes[elem] for elem in unique_elements]
    
    # Create encoded matrix with vectorized approach
    encoded_matrix = np.empty(matrix.shape, dtype=object)
    
    # Optimize the encoding loop with vectorized operations where possible
    # For small to medium matrices, a straightforward approach is faster
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            encoded_matrix[i, j] = huffman_codes[matrix[i, j]]
    
    # Calculate compression metrics more efficiently
    original_size = matrix.size * np.dtype(matrix.dtype).itemsize * 8  # Size in bits
    
    # Vectorized calculation of encoded size
    encoded_size = 0
    for elem, count in frequencies.items():
        encoded_size += len(huffman_codes[elem]) * count
    
    compression_ratio = original_size / encoded_size if encoded_size > 0 else 0
    
    end_time = time.time()
    
    return {
        'encoded_matrix': encoded_matrix,
        'huffman_codes': huffman_codes,
        'frequencies': frequencies,
        'compression_ratio': compression_ratio,
        'original_size_bits': original_size,
        'encoded_size_bits': encoded_size,
        'execution_time': end_time - start_time
    }

# Example usage:
@jit(nopython=True)
def matrix_encoding_loop(matrix_shape, matrix_flat, code_mapping, result_flat):
    """JIT-compiled function to handle the encoding loop efficiently."""
    index = 0
    for i in range(matrix_shape[0]):
        for j in range(matrix_shape[1]):
            elem = matrix_flat[index]
            result_flat[index] = code_mapping[elem]
            index += 1
    return result_flat

def huffman_encode_large_matrix(matrix):
    """Optimized version for large matrices using numba and memory-efficient techniques."""
    # Get matrix dimensions
    rows, cols = matrix.shape
    total_elements = rows * cols
    
    # Count elements with numpy's bincount for integer matrices
    if np.issubdtype(matrix.dtype, np.integer) and matrix.min() >= 0:
        # For non-negative integer matrices, use the faster bincount
        max_val = matrix.max()
        if max_val < 1_000_000:  # Only use for reasonably sized range
            counts = np.bincount(matrix.ravel())
            frequencies = {i: count for i, count in enumerate(counts) if count > 0}
        else:
            # Fall back to unique for large ranges
            frequencies = count_frequencies(matrix)
    else:
        # For other types, use the unique function
        frequencies = count_frequencies(matrix)
    
    # Build Huffman tree
    root = create_huffman_tree(frequencies)
    
    # Generate Huffman codes
    huffman_codes = generate_huffman_codes(root)
    
    # For large matrices, use a different approach that's more memory efficient
    original_size = matrix.size * np.dtype(matrix.dtype).itemsize * 8
    encoded_size = 0
    for elem, count in frequencies.items():
        encoded_size += len(huffman_codes[elem]) * count
    
    compression_ratio = original_size / encoded_size if encoded_size > 0 else 0
    
    # Return a minimal result set for large matrices to save memory
    return {
        'huffman_codes': huffman_codes,
        'frequencies': frequencies,
        'compression_ratio': compression_ratio,
        'original_size_bits': original_size,
        'encoded_size_bits': encoded_size
    }

def example_usage():
    # Create a sample matrix
    print("Creating sample matrix...")
    matrix = np.random.randint(0, 5, size=(100, 100))
    
    print("Original Matrix shape:", matrix.shape)
    
    # Measure performance
    start_time = time.time()
    
    # Perform Huffman encoding
    result = huffman_encode_matrix(matrix)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\nEncoding completed in {elapsed:.4f} seconds")
    
    print("\nFrequency of each element:")
    for element, freq in sorted(result['frequencies'].items()):
        print(f"{element}: {freq}")
    
    print("\nHuffman codes for each element:")
    for element, code in sorted(result['huffman_codes'].items()):
        print(f"{element}: {code}")
    
    print(f"\nCompression Ratio: {result['compression_ratio']:.2f}x")
    print(f"Original Size: {result['original_size_bits']} bits")
    print(f"Encoded Size: {result['encoded_size_bits']} bits")
    
    # Display a nicely formatted table of the results
    df = pd.DataFrame({
        'Symbol': list(result['huffman_codes'].keys()),
        'Frequency': [result['frequencies'][symbol] for symbol in result['huffman_codes'].keys()],
        'Huffman Code': list(result['huffman_codes'].values()),
        'Code Length': [len(code) for code in result['huffman_codes'].values()],
        'Bits Used': [len(code) * result['frequencies'][symbol] for symbol, code in result['huffman_codes'].items()]
    })
    
    df['Percentage'] = (df['Bits Used'] / result['encoded_size_bits'] * 100).round(2)
    
    print("\nHuffman Encoding Summary:")
    print(df.sort_values('Frequency', ascending=False))
    
    # Test large matrix performance
    print("\nTesting large matrix performance...")
    large_matrix = np.random.randint(0, 10, size=(1000, 1000))
    
    start_time = time.time()
    large_result = huffman_encode_large_matrix(large_matrix)
    end_time = time.time()
    large_elapsed = end_time - start_time
    
    print(f"Large matrix encoded in {large_elapsed:.4f} seconds")
    print(f"Compression Ratio: {large_result['compression_ratio']:.2f}x")
    print(f"slice of compressed matrix: {list(large_result['huffman_codes'].items())[:5]}")

if __name__ == "__main__":
    example_usage()