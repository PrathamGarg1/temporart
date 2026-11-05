"""Export to ONNX and dynamically quantize to INT8 (same recipe family as the shipped extension model).
  thread : fine-tuned 5-class thread classifier  (input_ids, attention_mask, token_type_ids -> logits)
  encoder: frozen base MiniLM + mean pooling      (input_ids, attention_mask -> 384-d vector), for the GRU
"""
import sys, warnings
from pathlib import Path
import torch
from transformers import AutoModel, AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "onnx"; OUT.mkdir(exist_ok=True)
BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Pooled(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask):
        h = self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mk = attention_mask.unsqueeze(-1).to(h.dtype)
        return (h * mk).sum(1) / mk.sum(1).clamp(min=1e-9)


def export(model, args, names, out, dyn):
    model.eval()
    torch.onnx.export(model, args, str(out), input_names=names, output_names=["out"],
                      dynamic_axes=dyn, opset_version=17, dynamo=False)


def main():
    ids = torch.randint(5, 1000, (2, 24)); am = torch.ones_like(ids); tt = torch.zeros_like(ids)
    which = sys.argv[1:] or ["thread", "encoder"]
    if "thread" in which:
        m = AutoModelForSequenceClassification.from_pretrained(ROOT / "artifacts_v2/cloud_results/model_thread")
        dyn = {k: {0: "b", 1: "s"} for k in ("input_ids", "attention_mask", "token_type_ids")} | {"out": {0: "b"}}
        export(m, (ids, am, tt), ["input_ids", "attention_mask", "token_type_ids"], OUT / "thread_fp32.onnx", dyn)
        quantize_dynamic(OUT / "thread_fp32.onnx", OUT / "thread_int8.onnx", weight_type=QuantType.QInt8)
    if "encoder" in which:
        m = Pooled(AutoModel.from_pretrained(BASE))
        dyn = {k: {0: "b", 1: "s"} for k in ("input_ids", "attention_mask")} | {"out": {0: "b"}}
        export(m, (ids, am), ["input_ids", "attention_mask"], OUT / "encoder_fp32.onnx", dyn)
        quantize_dynamic(OUT / "encoder_fp32.onnx", OUT / "encoder_int8.onnx", weight_type=QuantType.QInt8)
    for p in sorted(OUT.glob("*.onnx")):
        print(f"{p.name}: {p.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
