using Mmap
using CUDA

struct Config
    dim::Int
    h_dim::Int
    n_layers::Int
    n_heads::Int
    n_kv_heads::Int
    vocab_size::Int
    seq_len::Int
    buffer_layers::Int
end

struct Layer{T<:AbstractFloat}
    wq::Matrix{T}
    wk::Matrix{T}
    wv::Matrix{T}
    wo::Matrix{T}
    w1::Matrix{T}
    w2::Matrix{T}
    w3::Matrix{T}
    rms_att_weight::Matrix{T}
    rms_ffn_weight::Matrix{T}
    key_cache::Matrix{T,3}
    value_cache::Matrix{T,3}
end

struct LayerGPU{T<:AbstractFloat}
    wq::CuArray{T}
    wk::CuArray{T}
    wv::CuArray{T}
    wo::CuArray{T}
    w1::CuArray{T}
    w2::CuArray{T}
    w3::CuArray{T}
    rms_att_weight::CuArray{T}
    rms_ffn_weight::CuArray{T}
    key_cache::CuArray{T}
    value_cache::CuArray{T}
    LayerGPU(c::Config) = new(
        CuArray{T}(undef, c.dim, c.dim),
        CuArray{T}(undef, c.dim, c.dim),
        CuArray{T}(undef, c.dim, c.dim),
        CuArray{T}(undef, c.dim, c.dim),
        CuArray{T}(undef, c.dim, c.h_dim),
        CuArray{T}(undef, c.h_dim, c.dim),
        CuArray{T}(undef, c.dim, c.h_dim),
        CuArray{T}(undef, c.dim, 1),
        CuArray{T}(undef, c.dim, 1),
        CuArray{T}(undef, c.dim, c.seq_len)
    )
end


function Layer(f::IOStream, c::Config) where {T<:AbstractFloat}
    wq = mmap(f, Matrix{T}, (c.dim, c.dim))
    skip(f, sizeof(wq))
    wk = mmap(f, Matrix{T}, (c.dim, c.dim))
    skip(f, sizeof(wk))
    wv = mmap(f, Matrix{T}, (c.dim, c.dim))
    skip(f, sizeof(wv))
    wo = mmap(f, Matrix{T}, (c.dim, c.dim))
    skip(f, sizeof(wo))
    w1 = mmap(f, Matrix{T}, (c.dim, c.h_dim))
    skip(f, sizeof(w1))
    w2 = mmap(f, Matrix{T}, (c.h_dim, c.dim))
    skip(f, sizeof(w2))
    w3 = mmap(f, Matrix{T}, (c.dim, c.h_dim))
    skip(f, sizeof(w3))
    rms_att_weight = mmap(f, Matrix{T}, (c.dim, 1))
    skip(f, sizeof(rms_att_weight))
    rms_ffn_weight = mmap(f, Matrix{T}, (c.dim, 1))
    skip(f, sizeof(rms_ffn_weight))
    key_cache = zeros(T, c.dim, c.seq_len)
    value_cache = zeros(T, c.dim, c.seq_len)
    Layer{T}(wq, wk, wv, wo, w1, w2, w3, rms_att_weight, rms_ffn_weight, key_cache, value_cache)
end


struct Llama{T<:AbstractFloat}
    tok_embeddings::Matrix{T}
    rms_final_weight::Matrix{T}
    out_weights::Matrix{T}
    freq::Matrix{T}
    layers::Vector{Layer{T}}
end

function Llama(f::IOStream, c::Config) where {T<:AbstractFloat}
    tok_embedding = mmap(f, Matrix{T}, (c.dim, c.vocab_size))
    skip(f, sizeof(tok_embedding))
    rms_final_weight = mmap(f, Matrix{T}, (c.dim, 1))
    skip(f, sizeof(rms_final_weight))
    out_weights = mmap(f, Matrix{T}, (c.dim, c.vocab_size))
    skip(f, sizeof(out_weights))
    freq = mmap(f, Matrix{T}, (c.dim ÷ c.n_heads) ÷ 2, 1)
    skip(f, sizeof(freq))
    layers = [Layer(f, c) for _ in 1:32]
    Llama(tok_embedding, rms_final_weight, out_weights, freq, layers)
end

function copyto!(layer_gpu::LayerGPU{T}, layer_cpu::Layer{T}) where {T<:AbstractFloat}
    for field in fieldnames(layer_cpu)
        copyto!(getfield(layer_gpu, field), getfield(layer_cpu, field))
    end
    layer_gpu
end


struct GPUBuffer{T<:AbstractFloat}
    layers::Vector{LayersGPU{T}}
    x::CuArray{T}
    xb::CuArray{T}
    xb2::CuArray{T}
    hb::CuArray{T}
    hb2::CuArray{T}
    q::CuArray{T}
    k::CuArray{T}
    v::CuArray{T}
    att::CuArray{T}
    logits::CuArray{T}
end

function GPUBuffer(c::Config) where {T<:AbstractFloat}
    layers = [LayerGPU{T}(c) for _ in 1:2*c.buffer_layers]
    x = CuArray{T}(undef, c.dim)
    xb = CuArray{T}(undef, c.dim)
    xb2 = CuArray{T}(undef, c.dim)
    hb = CuArray{T}(undef, c.dim)
    hb2 = CuArray{T}(undef, c.dim)
    q = CuArray{T}(undef, c.dim)
    k = CuArray{T}(undef, c.dim)
    v = CuArray{T}(undef, c.dim)
    att = CuArray{T}(undef, c.seq_len)
    logits = CuArray{T}(undef, c.seq_len)
    GPUBuffer(layers, x, xb, xb2, hb, hb2, q, k, v, att, logits)
end
