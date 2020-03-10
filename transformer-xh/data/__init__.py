# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# import the classes in the files of the data folder here.
from .base import TransformerXHDataset
from .hotpotqa import HotpotDataset, batcher_hotpot
from .fever import FEVERDataset,batcher_fever
from .utils import load_data