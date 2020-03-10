# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

### Hotpot QA Eval:
#python transformer-xh/main.py --cf transformer-xh/configs/config_hotpot.json --test
### Hotpot QA Eval fp16:
python transformer-xh/main.py --cf transformer-xh/configs/config_hotpot.json --fp16 --test
