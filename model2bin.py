import click
import torch
from tqdm import trange


@click.command()
@click.option("--src")
@click.option("--tgt")
@click.option("--layer_cnt", default=32)
def convert(src, tgt, layer_cnt):
    model = torch.load(src)
    with open(tgt, "wb") as f:
        for name in ["tok_embeddings.weight", "norm.weight", "output.weight", "rope.freqs"]:
            f.write(model[name].view(-1).to(torch.float32).numpy())
        for layer_id in trange(layer_cnt):
            for name in [
                "attention.wq.weight",
                "attention.wk.weight",
                "attention.wv.weight",
                "attention.wo.weight",
                "feed_forward.w1.weight",
                "feed_forward.w2.weight",
                "feed_forward.w3.weight",
                "attention_norm.weight",
                "ffn_norm.weight",
            ]:
                f.write(model[f"layers.{layer_id}.{name}"].view(-1).to(torch.float32).numpy())


convert()
