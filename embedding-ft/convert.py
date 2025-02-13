from sentence_transformers import SentenceTransformer, models
from transformers.trainer import *
import argparse


def save_ckpt_for_sentence_transformers(
    ckpt_dir, out_dir, pooling_mode: str = "cls", normlized: bool = True
):
    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode
    )
    if normlized:
        normlize_layer = models.Normalize()
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model, normlize_layer], device="cpu"
        )
    else:
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device="cpu"
        )
    model.save(out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetuned Model.")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        help="Dir to the model finetuned.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Dir to save the model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_ckpt_for_sentence_transformers(args.ckpt_dir, args.out_dir)
