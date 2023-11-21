# GPT-J alphabetical probe

Experimental code which trains 26 linear probes to detect the presence of alphabetic letters in GPT-J token strings, given their embeddings. Exploring the resulting vector arithmetic and its impact on GPT-J spelling abilities

## Local Setup

```bash
conda create --name alphabet python=3.11
conda activate alphabet
pip install -r requirements.txt
```

## Vast.ai Setup

Go to vast.ai and hire an instance. Recommended 30+ GB of GPU RAM - I got a 48 GB one for about 50c/hr.
Open the instance.
Open a terminal (File -> New -> Terminal), and run the following set of commands:

```bash
git clone REPO_HERE # Or git clone -b BRANCH_HERE REPO_HERE
cd alphabetical_probe
conda create --name alphabet python=3.11
source activate alphabet
pip install -r requirements.txt
python -m ipykernel install --user --name=gpu
```

To run a notebook, you'll then need to navigate to the notebook, open it, and change the kernel to 'gpu' in the upper-right hand corner. Then you should just be able to run it as is!
