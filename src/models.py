from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from open_clip import create_model_and_transforms


MODEL_ALIASES: Dict[str, str] = {
    "echo_clip": "hf-hub:mkaichristensen/echo-clip",
    "biomed_clip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
}


@dataclass
class ModelConfig:
    model: str = "echo_clip"  # alias or full model id
    device: str = "cuda"
    precision: str = "bf16"


def resolve_model_id(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def load_model(cfg: ModelConfig):
    model_id = resolve_model_id(cfg.model)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_id,
        precision=cfg.precision,
        device=cfg.device,
    )
    model.eval()
    return model, preprocess_train, preprocess_val, model_id


def load_model_by_name(name: str, *, device: str, precision: str):
    cfg = ModelConfig(model=name, device=device, precision=precision)
    return load_model(cfg)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Load Echo/BioMed CLIP models")
    parser.add_argument("--model", default="echo_clip", choices=sorted(MODEL_ALIASES.keys()) + ["custom"], help="Alias to load or 'custom'")
    parser.add_argument("--model_id", default=None, help="Custom model id when --model=custom")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    args = parser.parse_args()

    model_name = args.model_id if args.model == "custom" else args.model
    model, _, _, resolved_id = load_model_by_name(model_name, device=args.device, precision=args.precision)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {resolved_id}")
    print(f"Device: {args.device} | Precision: {args.precision}")
    print(f"Total parameters: {total_params:,}")


if __name__ == "__main__":
    main()


__all__ = ["ModelConfig", "load_model", "load_model_by_name", "resolve_model_id", "MODEL_ALIASES"]
