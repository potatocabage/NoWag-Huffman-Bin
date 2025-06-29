import numpy as np
import heapq
from collections import Counter
import pandas as pd

class Node:
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
    nodes = []
    
    # Convert frequency dictionary to a list of Node objects
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
        new_node = Node(left.freq + right.freq, left.symbol + right.symbol, left, right)
        
        heapq.heappush(nodes, new_node)
    
    return nodes[0]

def generate_huffman_codes(root, code="", mapping=None):
    """Generate Huffman codes for each symbol based on the tree."""
    if mapping is None:
        mapping = {}
    
    # If leaf node, assign the current code to the symbol
    if not root.left and not root.right:
        mapping[root.symbol] = code
        return mapping
    
    # Traverse left with code + '0'
    if root.left:
        generate_huffman_codes(root.left, code + root.left.huff, mapping)
    
    # Traverse right with code + '1'
    if root.right:
        generate_huffman_codes(root.right, code + root.right.huff, mapping)
    
    return mapping

def huffman_encode_matrix(matrix):
    """Perform Huffman encoding on the elements of a matrix."""
    # Convert matrix to a 1D array for easier counting
    flattened = matrix.flatten()
    
    # Count frequency of each element
    frequencies = Counter(flattened)
    
    # Build Huffman tree
    root = create_huffman_tree(frequencies)
    
    # Generate Huffman codes
    huffman_codes = generate_huffman_codes(root)
    
    # Create encoded matrix (replace values with their Huffman codes)
    encoded_matrix = np.empty(matrix.shape, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            encoded_matrix[i, j] = huffman_codes[matrix[i, j]]
    
    # Calculate compression metrics
    original_size = matrix.size * np.dtype(matrix.dtype).itemsize * 8  # Size in bits
    encoded_size = sum(len(huffman_codes[element]) for element in flattened)
    compression_ratio = original_size / encoded_size if encoded_size > 0 else 0
    
    return {
        'encoded_matrix': encoded_matrix,
        'huffman_codes': huffman_codes,
        'frequencies': dict(frequencies),
        'compression_ratio': compression_ratio,
        'original_size_bits': original_size,
        'encoded_size_bits': encoded_size
    }

# Example usage:
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
    
    # Perform Huffman encoding
    result = huffman_encode_matrix(matrix)
    
    print("\nFrequency of each element:")
    for element, freq in result['frequencies'].items():
        print(f"{element}: {freq}")
    
    print("\nHuffman codes for each element:")
    for element, code in result['huffman_codes'].items():
        print(f"{element}: {code}")
    
    print("\nEncoded Matrix:")
    print(result['encoded_matrix'])
    
    print(f"\nCompression Ratio: {result['compression_ratio']:.2f}x")
    print(f"Original Size: {result['original_size_bits']} bits")
    print(f"Encoded Size: {result['encoded_size_bits']} bits")
    
    # Display a nicely formatted table of the results
    df = pd.DataFrame({
        'Symbol': list(result['huffman_codes'].keys()),
        'Frequency': [result['frequencies'][symbol] for symbol in result['huffman_codes'].keys()],
        'Huffman Code': list(result['huffman_codes'].values()),
        'Code Length': [len(code) for code in result['huffman_codes'].values()]
    })
    
    print("\nHuffman Encoding Summary:")
    print(df.sort_values('Frequency', ascending=False))

if __name__ == "__main__":
    example_usage()