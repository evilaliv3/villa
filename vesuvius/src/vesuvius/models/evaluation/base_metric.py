from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import numpy as np


def binarize_scores(scores: np.ndarray) -> np.ndarray:
    """Class ids from a single channel score volume: logits are cut at 0, probabilities at 0.5."""
    cutoff = 0.0 if (scores.min() < 0.0 or scores.max() > 1.0) else 0.5
    return (scores > cutoff).astype(np.int64)


class BaseMetric(ABC):
    def __init__(self, name: str):
        self.name = name
        self.results = []
    
    @abstractmethod
    def compute(self, pred: torch.Tensor, gt: torch.Tensor, **kwargs) -> Dict[str, float]:
        pass
    
    def update(self, pred: torch.Tensor, gt: torch.Tensor, **kwargs):
        result = self.compute(pred, gt, **kwargs)
        self.results.append(result)
        return result
    
    def aggregate(self) -> Dict[str, float]:
        if not self.results:
            return {}
        
        aggregated = {}
        all_keys = set()
        for result in self.results:
            all_keys.update(result.keys())
        
        for key in all_keys:
            values = [r[key] for r in self.results if key in r]
            if values:
                aggregated[key] = np.mean(values)
        
        return aggregated
    
    def reset(self):
        self.results = []