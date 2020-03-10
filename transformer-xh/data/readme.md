
# Data 
The data folder contains Dataset Class, Dataloader and necessary data vectorization functions (similar as BERT)

The base.py file contains the root dataset class (TransformerXHDataset).

For our experimented tasks (Hotpot QA and FEVER), each file contains the task specific class inherited from the root class. Similarily, you could create new files inherited from the root class for your task.
