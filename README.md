# Microsoft Open Source Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

Resources:

- [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)
- [Microsoft Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
- Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions or concerns


# Transformer-XH
The source codes of the paper "Transformer-XH: Multi-evidence Reasoning with Extra Hop Attention (ICLR 2020)".

# Dependency Installation
Run python setup.py develop to install required dependencies for transformer-xh.

# Data and trained model Download

You can run bash script download.sh

For Hotpot QA, we provide processed graph (Transformer-XH) input [here](https://1drv.ms/u/s!AgvaHirT432UgRj391A5YGaQAk7i?e=eCeHFm), after downloading, unzip it and put into ./data folder
We also provide trained model [here](https://1drv.ms/u/s!AgvaHirT432UgRXigpatnZAKjMBN?e=vay34o), unzip the downloaded model and put into ./experiments folder

Similarly, we provide processed graph in fever [here](https://1drv.ms/u/s!AgvaHirT432UgRadjowRVd_TQuwS?e=Qcdxse), and trained model [here](https://1drv.ms/u/s!AgvaHirT432UgReDSqnayo8mjW0M?e=4KoJtn). 


## Run Your Models
Use hotpot_train.sh for training on hotpot QA task, hotpot_eval.sh for evaluation (default fp16 training).

Similarly, fever_train.sh for training on FEVER task, fever_eval.sh for evaluation (default fp16 training).


## Contact
If you have questions, suggestions and bug reports, please email chenz@cs.umd.edu.