
# pegasus-x-cord19

This repository is one of the main contributions of the research paper entitled "Attention to COVID-19: Abstractive Summarization of COVID-19 Research with State-of-the-Art Transformers", authored by Jan Apolline D. Estrella, Christian S. Quinzon, Dr. Francis George C. Cabarle and Dr. Jhoirene B. Clemente of the Algorithms and Complexity Laboratory at the University of the Philippines - Diliman. 

We hope to provide a replicable experimental setup for future researchers who are interested in learning how to finetune Hugging Face transformers for specific tasks, like summarization.







## Checkpoints

In our research, we finetuned PEGASUS-X-BASE on the CORD-19 dataset for the specific task of summarizing COVID-19 research.

Our PEGASUS-X generated checkpoints are publicly available on Hugging Face 🤗:

- [PEGASUS-X-CORD-19](https://huggingface.co/aplnestrella/pegasus-x-cord19)
- [PEGASUS-X-arXiv-CORD-19](https://huggingface.co/aplnestrella/pegasus-x-arXiv-cord19)
## Setup

Use `finetuning/finetune.py` to finetune your Hugging Face model to generate a new checkpoint). The set hyperparameters allow finetuning to be feasible even with a single GPU. In our case, we utilized one NVIDIA A100 80 GB Tensor Core GPU.

Use `evaluating/evaluate.py` to evaluate your checkpoint on a specific task (e.g., summarization).

