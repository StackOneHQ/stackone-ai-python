# Hybrid search default weight for BM25 vs TF-IDF
# alpha=0.2 means: 20% BM25 + 80% TF-IDF
# This value was optimized through validation testing and provides
# 10.8% improvement in tool discovery accuracy
DEFAULT_HYBRID_ALPHA: float = 0.2
