"""Translate a UTF-8 text file with the downloaded local NLLB model."""
import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    if not args.input.is_file():
        p.error(f"Input file missing: {args.input}")
    if args.output.exists():
        p.error(f"Output exists; choose a new path: {args.output}")
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_dir = ROOT / "checkpoints" / "nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(model_dir, src_lang="eng_Latn", local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True).to(args.device).eval()
    text = args.input.read_text(encoding="utf-8-sig")
    paragraphs = [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]
    # Segment at sentence boundaries and reject overlong sentences, never silently truncate.
    blocks = []
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        chunks, current = [], ""
        for sentence in sentences:
            if len(tok(sentence)["input_ids"]) > 450:
                raise ValueError("A sentence exceeds 450 tokens. Split it into shorter sentences first.")
            candidate = (current + " " + sentence).strip()
            if len(tok(candidate)["input_ids"]) > 450:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
        blocks.append(chunks)
    output = []
    with torch.inference_mode():
        for chunks in blocks:
            translated = []
            for chunk in chunks:
                batch = tok(chunk, return_tensors="pt").to(args.device)
                result = model.generate(**batch, forced_bos_token_id=tok.convert_tokens_to_ids("vie_Latn"), max_new_tokens=512, num_beams=4)
                if result.shape[1] >= 513 and result[0, -1].item() != tok.eos_token_id:
                    raise RuntimeError("Translation reached the token limit. Use shorter sentences.")
                translated.append(tok.decode(result[0], skip_special_tokens=True))
            output.append(" ".join(translated))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(output), encoding="utf-8")
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
