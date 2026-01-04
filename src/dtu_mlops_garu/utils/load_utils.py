from pathlib import Path

def find_model_path(model_checkpoint: str) -> str:
    p = Path(model_checkpoint)
    if not p.exists():
        p_alt = Path("/") / model_checkpoint.lstrip("/")
        if p_alt.exists():
            p = p_alt
    if not p.exists():
        raise FileNotFoundError(f"Model checkpoint not found: tried {model_checkpoint} and {p}")
    
    return p