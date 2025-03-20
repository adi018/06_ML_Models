import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        """
        Params:
            d_model: int, dimension of model
            vocab_size: int, size of vocabulary
        Returns:
            None
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.vocab_size = vocab_size
    
    def forward(self, x):
        """
        Params:
            x: torch.tensor, shape [batch_size, seq_len]
        Returns:
            torch.tensor, shape [batch_size, seq_len, d_model]
        """
        
        return self.embeddings(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Params:
            d_model: int, dimension of model. Size of the vector that represents each token
            dropout: float, dropout rate. to make the model less overfitting
            seq_len: int, maximum length of sequence
        Returns:
            None
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        # To create positional encoding a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector to represent the position of word in a sentence of shape (seq_len, 1)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Even and Odd positional encoding for each word in a sentence
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension to positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # Register the positional encoding: When we want to keep a tensor inside the model not as a learned parameter
        # but as a constant tensor and save the tensor when the model is saved, we can register it as a buffer.
        # So, when we save the model, the buffer will be saved as well.
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Params:
            x: torch.tensor, shape [batch_size, seq_len, d_model]
        Returns:
            torch.tensor, shape [batch_size, seq_len, d_model]
        """
        # Add Positional Encoding to every word embedding in a sentence
        # But as positional encoding is not a learned parameter, we need to detach it from the computation graph
        # and set the requires_grad to False
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)

        # dropout the summed embeddings and positional encoding
        # reason: to make the model less overfitting and relying heavily on a given positional encoding
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    # Layer Normalization is different from Batch Normalization
    # How? Batch Normalization normalizes the activations of each layer in a mini-batch
    # while Layer Normalization normalizes the activations of each layer in the model
    # independently of the other layers
    
    def __init__(self, eps: float = 1e-6):
        """
        Params:
            eps: float, epsilon value to avoid division by zero
        Returns:
            None
        """
        super().__init__()
        self.eps = eps

        
        # Multiplied
        self.alpha = nn.Parameter(torch.ones(1)) # nn.Parameter: makes the paramerter learnable
        # Added
        self.bias = nn.Parameter(torch.zeros(1)) # nn.Parameter: makes the paramerter learnable
    
    def forward(self, x):
        """
        Params:
            x: torch.tensor, shape [batch_size, seq_len, d_model]
        Returns:
            torch.tensor, shape [batch_size, seq_len, d_model]
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        Params:
            d_model: int, dimension of model
            d_ff: int, dimension of feed forward layer
            dropout: float, dropout rate
        Returns:
            None
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        """
        Params:
            x: torch.tensor, shape [batch_size, seq_len, d_model]
        Returns:
            torch.tensor, shape [batch_size, seq_len, d_model]
        """
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    # MultiHeadAttention is a mechanism that allows the model to focus on different parts of the input sequence
    # It is divided into three parts:
    # 1. Linear Transformation
    # 2. Split
    # 3. Scaled Dot-Product Attention   

    # In MultiHeadAttention, we have four weight matrices:
    # 1. W_q: Query matrix
    # 2. W_k: Key matrix
    # 3. W_v: Value matrix
    # 4. W_o: Output matrix

    # Multihead Attention is calculated as follows:
    # 1. Linear Transformation:
    #    Q´ = X * W_q
    #    K´ = X * W_k
    #    V´ = X * W_v
    # 2. Split:
    #    Q_{ki} = Split(Q´)
    #    K_{ki} = Split(K´)
    #    V_{ki} = Split(V´)
    # 3. Scaled Dot-Product Attention:
    #    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    # 4. Concatenation:
    #    Concat(Attention(Q, K, V)) = Concat(Attention(Q, K, V))
    # 5. Output Transformation:
    #    Output = Concat(Attention(Q, K, V)) * W_o

    # In multihead the Q´, K´, V´ are divided into h heads along the embedding dimension
    # and not along sequence direction
    # So, each head has access to the full sentence but only to a part of the embedding

    def __init__(self, d_model: int, h: int, dropout: float):
        """
        Params:
            d_model: int, dimension of model
            h: int, number of heads
            dropout: float, dropout rate
        Returns:
            None
        """
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h i.e. num of heads"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        # W_q, W_k, W_v, W_o
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores
    
    
    def forward(self, q, k, v, mask=None):
        """
        Params:
            q: torch.tensor, Query -> shape [batch_size, seq_len, d_model]
            k: torch.tensor, Key -> shape [batch_size, seq_len, d_model]
            v: torch.tensor, Value -> shape [batch_size, seq_len, d_model]
            mask: torch.tensor, shape [batch_size, seq_len]
        Returns:
            torch.tensor, shape [batch_size, seq_len, d_model]
        """
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        """
        Params:
            features: int, number of features
            dropout: float, dropout rate
        Returns:
            None
        """
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        """
        Params:
            x: torch.tensor, shape [batch_size, seq_len, d_model]
            sublayer: nn.Module, sublayer to apply i.e. the layer that was skipped by residual connection like MultiHeadAttentionBlock or FeedForwardBlock
        Returns:
            torch.tensor, shape [batch_size, seq_len, d_model]
        """
        return x + self.dropout(sublayer(self.norm(x))) # we could have also used self.dropout(self.norm(x + sublayer(x)))
    
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        Params:
            features: int, number of features
            self_attention_block: MultiHeadAttentionBlock, self attention block
            feed_forward_block: FeedForwardBlock, feed forward block
            dropout: float, dropout rate
        Returns:
            None
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """
        Params:
            x: torch.tensor, shape [batch_size, seq_len, d_model]
            src_mask: torch.tensor, shape [batch_size, seq_len]
        Returns:
            torch.tensor, shape [batch_size, seq_len, d_model]
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        Params:
            features: int, number of features
            layers: nn.ModuleList, list of encoder blocks
        Returns:
            None
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        """
        Params:
            x: torch.tensor, shape [batch_size, seq_len, d_model]
            mask: torch.tensor, shape [batch_size, seq_len]
        Returns:
            torch.tensor, shape [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        Params:
            features: int, number of features
            self_attention_block: MultiHeadAttentionBlock, self attention block
            cross_attention_block: MultiHeadAttentionBlock, cross attention block
            feed_forward_block: FeedForwardBlock, feed forward block
            dropout: float, dropout rate
        Returns:
            None
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Params:
            x: torch.tensor, shape [batch_size, seq_len, d_model]
            encoder_output: torch.tensor, shape [batch_size, seq_len, d_model]
            src_mask: torch.tensor, shape [batch_size, seq_len] coming from Encoder
            tgt_mask: torch.tensor, shape [batch_size, seq_len] coming from Decoder
        Returns:
            torch.tensor, shape [batch_size, seq_len, d_model]
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        Params:
            features: int, number of features
            layers: nn.ModuleList, list of decoder blocks
        Returns:
            None
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Params:
            x: torch.tensor, shape [batch_size, seq_len, d_model]
            encoder_output: torch.tensor, shape [batch_size, seq_len, d_model]
            src_mask: torch.tensor, shape [batch_size, seq_len]
            tgt_mask: torch.tensor, shape [batch_size, seq_len]
        Returns:
            torch.tensor, shape [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    # It is the linear layer that projects the output of the decoder to the vocabulary size
    # It is used to calculate the probability of each word in the vocabulary

    def __init__(self, d_model, vocab_size) -> None:
        """
        Params:
            d_model: int, dimension of model
            vocab_size: int, size of vocabulary
        Returns:
            None
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        """
        Params:
            x: torch.tensor, shape [batch_size, seq_len, d_model]
        Returns:
            torch.tensor, shape [batch_size, seq_len, vocab_size]
        """
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        """
        Params:
            encoder: Encoder, encoder block
            decoder: Decoder, decoder block
            src_embed: InputEmbeddings, source embeddings
            tgt_embed: InputEmbeddings, target embeddings
            src_pos: PositionalEncoding, source positional encoding
            tgt_pos: PositionalEncoding, target positional encoding
            projection_layer: ProjectionLayer, projection layer
        Returns:
            None
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        Params:
            src: torch.tensor, shape [batch_size, seq_len]
            src_mask: torch.tensor, shape [batch_size, seq_len]
        Returns:
            torch.tensor, shape [batch_size, seq_len, d_model]
        """
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        """
        Params:
            encoder_output: torch.tensor, shape [batch_size, seq_len, d_model]
            src_mask: torch.tensor, shape [batch_size, seq_len]
            tgt: torch.tensor, shape [batch_size, seq_len]
            tgt_mask: torch.tensor, shape [batch_size, seq_len]
        Returns:
            torch.tensor, shape [batch_size, seq_len, d_model]
        """
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        """
        Params:
            x: torch.tensor, shape [batch_size, seq_len, d_model]
        Returns:
            torch.tensor, shape [batch_size, seq_len, vocab_size]
        """
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    """
    Params:
        src_vocab_size: int, size of source vocabulary
        tgt_vocab_size: int, size of target vocabulary
        src_seq_len: int, maximum length of source sequence
        tgt_seq_len: int, maximum length of target sequence
        d_model: int, dimension of model
        N: int, number of encoder and decoder blocks
        h: int, number of heads
        dropout: float, dropout rate
        d_ff: int, dimension of feed forward layer
    Returns:
        Transformer
    """
    
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer