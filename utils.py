from functools import partial
from typing import Dict, Optional, Optional, Union
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from datetime import datetime

def get_timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M%S")

Data = Dict[str, Tensor]
DatasetType = Dataset[Data]
DataLoaderType = DataLoader[Data]
OptimizerType = Optional[partial[Optimizer]]
LRSchedulerType = Optional[dict[str, Union[partial[LRScheduler], str]]]