
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

# Model
The model folder contains Transformer-XH model, and task specifc layers.
The model.py file contains the root class (Model) and Transformer-XH component.
For our experimented tasks (Hotpot QA and FEVER), each file contains the task specific class inherited from the root class (Only the last layer is task specific). Similarily, you could create new files inherited from the root class for your task.
