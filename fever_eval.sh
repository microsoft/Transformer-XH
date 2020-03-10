
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

### FEVER Eval:
#python transformer-xh/main.py --cf transformer-xh/configs/config_fever.json --test
### FEVER Eval fp16:
python transformer-xh/main.py --cf transformer-xh/configs/config_fever.json --test --fp16
