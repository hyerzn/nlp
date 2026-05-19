from diversity.embedding import remote_clique, chamfer_dist

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A swift auburn fox vaulted a sleeping canine.",
    "I brewed coffee and read the paper."
]

# Remote Clique Score
rc = remote_clique(texts, model="Qwen/Qwen3-Embedding-0.6B")
print(f"Remote Clique: {rc:.3f}")

# Chamfer Distance
cd = chamfer_dist(texts, model="Qwen/Qwen3-Embedding-0.6B")
print(f"Chamfer Distance: {cd:.3f}")