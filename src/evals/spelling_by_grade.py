import random
from typing import Dict, List, Tuple

from evals.eval_utils import get_spelling, run_inference_on_model

def prepare_grade_spelling_eval(filename: str, separator: str, case='upper'):
    """Takes a file with words separated by grades. 
    Returns a dictionary of tuples (word, spelling) by grade.
    Format {1: [(word, spelling), (word, spelling), ...], 2: [...], ...}"""

    # Get the words and their correct spelling by grade from 1-5.
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()
        lines = [l.replace('\n', '').replace('*', '') for l in lines]
        
        # Determine which lines are in grade 1, 2, etc.
        grade_indices = [i for i in range(len(lines)-1) if lines[i].startswith("GRADE")] + [len(lines)]
        words_by_grade = {i+1: [] for i in range(len(grade_indices)-1)}
        
        for i, idx in enumerate(grade_indices):
            if i+1 < len(grade_indices): # Ensure we don't overflow.
                for l in range(idx+2, grade_indices[i+1] - 1): # Collect all words of the right grade.
                    words_by_grade[i+1].append((lines[l].strip(), get_spelling(lines[l].strip(), separator, case)))

    return words_by_grade


def create_few_shot_grade_spelling_prompt(word: str, word_list: Tuple[str, str], num_shots: int):
    """Takes in a word we want to spell, a list of words to sample from, and the number of shots.
    Tuple is of the form (word, spelling)."""
    assert 0 <= num_shots < len(word_list), "Number of shots must be between 0 and the number of words in the list, minus the chosen word."
    prompt = ''
    if word in word_list:
        word_list.remove(word) # Ensure we don't sample the same word we want to spell.
    if num_shots > 0:
        samples = random.sample(word_list, num_shots)
        for sample in samples:
            prompt += f"Q: How do you spell '{sample[0]}'? A: {sample[1]}\n\n"
    prompt += f"Q: How do you spell '{word}'? A:"
    return prompt


def assess_model_on_words(model, model_type: str, tokenizer, word_list: List[Tuple[str, str]], num_shots: int, batch_size=10):
    """Takes in our list of words and how many shots we want to give the model.
    Creates prompts using those few shots, then runs inference on the model."""
    data = {grade: [] for grade in word_list.keys()}
    
    for grade in word_list:
        print(f"Assessing Grade {grade}")
        prompts = [create_few_shot_grade_spelling_prompt(word[0], word_list[grade], num_shots) for word in word_list[grade]]
        data[grade] = run_inference_on_model(model, model_type, tokenizer, prompts, [w[1] for w in word_list[grade]], batch_size)

    return data

def get_spelling_accuracy(data: Dict[int, List[Dict[int, float]]]):
    """Takes in a dictionary of results, and returns the accuracy of the model on each grade."""
    return {grade: sum([1 if d['response'].startswith(d['answer']) else 0 for d in data[grade]]) / len(data[grade]) for grade in data.keys()}