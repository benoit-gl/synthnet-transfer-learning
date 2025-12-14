"""SwinV2 (Swin Transformer V2) module.

SwinV2 uses the same HuggingFace AutoModelForImageClassification interface as ViT,
so we reuse VitModule. The model architecture (shifted windows, hierarchical features)
is handled automatically by HuggingFace based on the model_name.

Note: If VitModule ever adds ViT-specific logic (e.g., accessing architecture-specific
layers), this alias may need to become a proper subclass or separate implementation.
"""

from models.vit_module import VitModule

SwinV2Module = VitModule
