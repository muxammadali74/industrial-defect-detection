import os
import warnings
import torch
from unittest.mock import patch
from anomalib.data import MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

class DummyTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        if self.iterable is not None:
            return iter(self.iterable)
        return self
    def __next__(self):
        return next(self.iterable)
    def update(self, *args, **kwargs):
        pass
    def close(self, *args, **kwargs):
        pass



datamodule = MVTecAD(
    root="/mvtec-ad",
    category="bottle",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=2,
)

model = Patchcore(
    num_neighbors=6,
)

engine = Engine(
    max_epochs=1,
    accelerator="auto",
)


with patch('anomalib.models.components.sampling.k_center_greedy.tqdm', DummyTqdm):
    engine.fit(datamodule=datamodule, model=model)

test_results = engine.test(datamodule=datamodule, model=model)
engine.export(
    model=model,
    export_type=ExportType.ONNX,
)
