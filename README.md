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

## Evals

`eval_list.py` in src/evals contains a list of evaluations, which also links to the `GradeSpellingEval` class that constructs them. To make a new evaluation, all you need to do is build your functions according to this class and add the evaluation to eval_list.py.

Instructions for how to load, save, and run evals can be found in `Eval_Demo.ipynb` under src/evals/notebooks.

## Probing

TBD

## Analysis

TBD
