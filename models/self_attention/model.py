import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerEncoder(layers.Layer):
    """
    Transformer Encoder block consists of multi-head self-attention and a feed-forward network.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Multi-head self-attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    """
    Encodes tokens and their positions into vectors.
    """
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(x)
        return token_embeddings + position_embeddings

def create_model(maxlen, vocab_size, embed_dim=128, num_heads=8, ff_dim=128, num_transformer_blocks=4, dropout_rate=0.1):
    """
    Creates the Transformer-based binary classification model (Head A).

    Args:
        maxlen (int): Maximum sequence length.
        vocab_size (int): The size of the vocabulary.
        embed_dim (int): The dimensionality of the embedding space.
        num_heads (int): The number of attention heads.
        ff_dim (int): The dimensionality of the feed-forward network.
        num_transformer_blocks (int): The number of Transformer Encoder blocks.
        dropout_rate (float): The dropout rate.

    Returns:
        keras.Model: The compiled Keras model.
    """
    inputs = layers.Input(shape=(maxlen,))
    
    # 1. Embedding Layer
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    
    # 2. Core Model: Stacked Transformer Encoders
    for _ in range(num_transformer_blocks):
        x = TransformerEncoder(embed_dim, num_heads, ff_dim, dropout_rate)(x)

    # 3. OUTPUT HEADS    
    # Head A: Binary Classification (Vulnerable / Not Vulnerable)
    pooled_output = layers.GlobalAveragePooling1D()(x)
    pooled_output = layers.Dropout(0.4)(pooled_output)
    pooled_output = layers.Dense(16, activation="relu")(pooled_output)
    pooled_output = layers.Dropout(0.4)(pooled_output)
    # Output layer for binary classification
    binary_output = layers.Dense(1, activation="sigmoid", name="binary_classification_head")(pooled_output)

    # Head B: Multi-Label Vulnerability Classification

    # Head C: Vulnerability Localization (Token-level)

    # --- Model Definition ---
    model = keras.Model(inputs=inputs, outputs=binary_output)
    
    return model

if __name__ == '__main__':
    VOCAB_SIZE = 30000  # Example: Size of your vocabulary
    MAX_LEN = 1024      # Example: Max length of input sequences
    EMBED_DIM = 128     # Embedding dimension
    NUM_HEADS = 4       # Number of attention heads
    FF_DIM = 128        # Feed-forward network dimension
    NUM_TRANSFORMER_BLOCKS = 4 # Number of encoder layers

    vulnerability_model = create_model(
        maxlen=MAX_LEN,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS
    )

    # Print a summary of the model architecture
    vulnerability_model.summary()

    # Compile the model for training
    vulnerability_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy", 
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    print("\nModel compiled successfully.")
