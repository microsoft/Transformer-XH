# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

### FEVER Train:
#python transformer-xh/main.py --cf transformer-xh/configs/config_fever.json 
### FEVER Train fp16:
python transformer-xh/main.py --cf transformer-xh/configs/config_fever.json --fp16
