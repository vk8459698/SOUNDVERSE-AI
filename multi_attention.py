import numpy as np

def split_heads(X, n_heads):
    """
    Split the last dimension of X into (n_heads, d_k/n_heads)
    
    Args:
        X: Input tensor of shape (seq_len, d_k)
        n_heads: Number of attention heads
    
    Returns:
        reshaped X with shape (n_heads, seq_len, d_k/n_heads)
    """
    seq_len, d_k = X.shape
    # Ensure d_k is divisible by n_heads
    assert d_k % n_heads == 0, "Feature dimension must be divisible by number of heads"
    
    d_head = d_k // n_heads
    
    # Reshape to (seq_len, n_heads, d_head)
    X_reshaped = X.reshape(seq_len, n_heads, d_head)
    
    # Transpose to (n_heads, seq_len, d_head)
    return X_reshaped.transpose(1, 0, 2)

def combine_heads(X):
    """
    Combine heads by transposing and reshaping.
    
    Args:
        X: Input tensor of shape (n_heads, seq_len, d_head)
    
    Returns:
        combined tensor of shape (seq_len, n_heads*d_head)
    """
    n_heads, seq_len, d_head = X.shape
    
    # Transpose back to (seq_len, n_heads, d_head)
    X_transposed = X.transpose(1, 0, 2)
    
    # Reshape to (seq_len, n_heads*d_head)
    return X_transposed.reshape(seq_len, n_heads * d_head)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Query matrix of shape (n_heads, seq_len, d_head)
        K: Key matrix of shape (n_heads, seq_len, d_head)
        V: Value matrix of shape (n_heads, seq_len, d_head)
        mask: Optional mask to apply to attention scores
    
    Returns:
        output: Attention output of shape (n_heads, seq_len, d_head)
    """
    # Get dimensions
    _, _, d_head = K.shape
    
    # Step 1: Compute attention scores (batched matrix multiplication)
    # attention_scores shape: (n_heads, seq_len, seq_len)
    attention_scores = np.matmul(Q, np.transpose(K, (0, 2, 1)))
    
    # Step 2: Scale attention scores
    attention_scores = attention_scores / np.sqrt(d_head)
    
    # Step 3: Apply mask if provided
    if mask is not None:
        attention_scores = attention_scores + mask
    
    # Step 4: Apply softmax to get attention weights
    # attention_weights shape: (n_heads, seq_len, seq_len)
    attention_weights = np.exp(attention_scores)
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    # Step 5: Compute weighted sum of values
    # output shape: (n_heads, seq_len, d_head)
    output = np.matmul(attention_weights, V)
    
    return output

def multi_head_attention(Q, K, V, n_heads):
    """
    Compute multi-head attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_model)
        K: Key matrix of shape (seq_len, d_model)
        V: Value matrix of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        output: Multi-head attention output of shape (seq_len, d_model)
    """
    # Step 1: Split into multiple heads
    # Q_split shape: (n_heads, seq_len, d_model/n_heads)
    Q_split = split_heads(Q, n_heads)
    K_split = split_heads(K, n_heads)
    V_split = split_heads(V, n_heads)
    
    # Step 2: Apply scaled dot-product attention to each head
    # attention_output shape: (n_heads, seq_len, d_model/n_heads)
    attention_output = scaled_dot_product_attention(Q_split, K_split, V_split)
    
    # Step 3: Combine the heads back
    # combined_output shape: (seq_len, d_model)
    combined_output = combine_heads(attention_output)
    
    return combined_output

# Example usage
if __name__ == "__main__":
    # Example input
    Q = np.array([[1, 0], [0, 1]])
    K = np.array([[1, 0], [0, 1]])
    V = np.array([[1, 0], [0, 1]])
    n_heads = 2
    
    # Apply multi-head attention
    output = multi_head_attention(Q, K, V, n_heads)
    
    print("Multi-Head Attention Output:")
    print(output)