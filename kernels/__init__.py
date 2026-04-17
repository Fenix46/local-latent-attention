# Triton kernels for LocalLatentAttention
try:
    from .llattn_op import LLAttnFunction
except ImportError:
    from llattn_op import LLAttnFunction

__all__ = ["LLAttnFunction"]
