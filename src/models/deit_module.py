"""DeiT (Data-efficient Image Transformer) module.

DeiT uses the same HuggingFace AutoModelForImageClassification interface as ViT,
so we reuse VitModule. The model architecture (including the distillation token)
is handled automatically by HuggingFace based on the model_name.

Note: If VitModule ever adds ViT-specific logic (e.g., accessing architecture-specific
layers), this alias may need to become a proper subclass or separate implementation.
"""

from models.vit_module import VitModule

DeitModule = VitModule
