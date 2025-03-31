import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """
    Compute query, key, and value matrices for self-attention.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_q: Query weight matrix of shape (d_model, d_k)
        W_k: Key weight matrix of shape (d_model, d_k)
        W_v: Value weight matrix of shape (d_model, d_v)
    
    Returns:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    """
    # Compute query, key, and value matrices by multiplying input with respective weights
    Q = np.dot(X, W_q)  # Shape: (seq_len, d_k)
    K = np.dot(X, W_k)  # Shape: (seq_len, d_k)
    V = np.dot(X, W_v)  # Shape: (seq_len, d_v)
    
    return Q, K, V

def self_attention(Q, K, V, mask=None, scale=True):
    """
    Compute self-attention mechanism.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        mask: Optional mask to apply to attention scores
        scale: Whether to scale attention scores by sqrt(d_k)
    
    Returns:
        output: Self-attention output of shape (seq_len, d_v)
    """
    # Get dimensions
    seq_len, d_k = K.shape
    
    # Step 1: Compute attention scores by multiplying Q with K transpose
    # Shape: (seq_len, seq_len)
    attention_scores = np.dot(Q, K.T)
    
    # Step 2: Scale attention scores by sqrt(d_k) for stable gradients
    if scale:
        attention_scores = attention_scores / np.sqrt(d_k)
    
    # Step 3: Apply mask if provided (used in decoder for causal attention)
    if mask is not None:
        attention_scores = attention_scores + mask
    
    # Step 4: Apply softmax to get attention weights
    # Shape: (seq_len, seq_len)
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)
    
    # Step 5: Compute weighted sum of values based on attention weights
    # Shape: (seq_len, d_v)
    output = np.dot(attention_weights, V)
    
    return output

# Example usage
if __name__ == "__main__":
    # Example input
    X = np.array([[1, 0], [0, 1]])
    W_q = np.array([[1, 0], [0, 1]])
    W_k = np.array([[1, 0], [0, 1]])
    W_v = np.array([[1, 2], [3, 4]])
    
    # Compute query, key, and value matrices
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    
    # Apply self-attention
    output = self_attention(Q, K, V)
    
    print("Self-Attention Output:")
    print(output)