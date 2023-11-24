from evals.eval_utils import get_spelling, run_inference_on_model, EvalResponse, ModelType
from typing import Dict, List, Tuple, TypedDict

import random
import re

# TODO: Remove tokenizer, use model.tokenizer if possible.
# TODO: Add "is letter in word" eval.
# Should this be in eval_utils?
class SpellingEvalDict(TypedDict):
    word: str
    prompt: str
    answer: str
    response: str
    

class EvalResponseDict(TypedDict):
    data: Dict[int, List[SpellingEvalDict]]
    accuracy: Dict[int, float]
    
    
# TODO: Consider an Eval generic class that can be extended for different evals.
class SpellingEval:
    """An object that contains the functions needed to run an eval on spelling by grade.
    Together, the functions define the evaluation.
    
    args:
    prompt_function: A function that takes in a word, a list of words to sample from, and the number of shots.
    Returns a prompt for the model to answer.
    format_response_function: A function that takes in the response from the model and formats it as needed.
    metric_function: A function that takes in a dictionary of {grade: [{prompt: str, answer: str, response: str}, ...], ...
    and judges the accuracy of the responses based on the answers."""
    def __init__(self, prompt_function: callable, format_response_function: callable, metric_function: callable):
        self.prompt_function = prompt_function
        self.format_response_function = format_response_function
        self.metric_function = metric_function
    
    def create_prompt(self, word: str, word_list: Tuple[str, str], num_shots: int) -> Tuple[str, str]:
        """Create a prompt for the model to answer.
        Returns the prompt and the answer to the prompt."""
        return self.prompt_function(word, word_list, num_shots)
    
    def format_response(self, response: str) -> str:
        """Format the responses the model gives as needed."""
        return self.format_response_function(response)
    
    def get_accuracy(self, data: Dict[int, List[SpellingEvalDict]]) -> Dict[int, float]:
        """Takes in a dictionary of {grade: [{prompt: str, answer: str, response: str}, ...], ...
        Returns the accuracy of the model on each grade according to the eval's metric function."""
        return self.metric_function(data)
    
    def run_eval(self, model, model_type: ModelType, tokenizer, 
                 word_list: List[Tuple[str, str]], 
                 num_shots: int, batch_size: int=10) -> EvalResponseDict:
        """Runs the eval on a given word list and number of shots.
        
        args:
        model: Contains the HuggingFace or TransformerLens model to run inference on.
        model_type: Determines what type of model we should use, which determines how we run inference.
        tokenizer: Contains the tokenizer to apply to prompts.
        word_list: A list of tuples of the form (word, spelling) to run the eval on.
        num_shots: How many shots to give the model in evaluation.
        batch_size: How many prompts to pass in at once to the model.
        
        Returns:
        An EvalResponse Dict containing the data and accuracy of the eval.
        """
        data = {grade: [] for grade in word_list.keys()}
    
        for grade in word_list:
            print(f"Assessing Grade {grade}")
            prompts, answers = zip([self.prompt_function(word[0], word_list[grade], num_shots) for word in word_list[grade]])
            data[grade] = run_inference_on_model(model, model_type, tokenizer, prompts, answers, batch_size)
 
        response: EvalResponseDict = {'data': data, 'accuracy': self.get_accuracy(data)}
        return response
        
    
# Full spelling eval
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


def create_full_spelling_prompt(word: str, word_list: Tuple[str, str], num_shots: int) -> Tuple[str, str]:
    """Takes in a word we want to spell, a list of words to sample from, and the number of shots.
    Tuple is of the form (word, spelling).
    Creates a prompt that asks for the full spelling of a word.
    
    Returns the prompt and the answer to the prompt."""
    assert 0 <= num_shots < len(word_list), "Number of shots must be between 0 and the number of words in the list, minus the chosen word."
    assert word in word_list, "Word must be in the word list."
    prompt = ''
    word_list.remove(word)
    if num_shots > 0:
        samples = random.sample(word_list, num_shots)
        for sample in samples:
            prompt += f"Q: How do you spell '{sample[0]}'? A: {sample[1]}\n\n"
    prompt += f"Q: How do you spell '{word}'? A:"
    return prompt, word_list[word][1]


def format_full_spelling_response(item):
    """Format the response to a full spelling prompt."""
    return item['response'].split('A: ')[-1].split("\n\n")[0].replace('\n', '').replace('<|endoftext|>', '').strip()


# First letter eval
def create_first_letter_spelling_prompt(word: str, word_list: Tuple[str, str], num_shots: int) -> Tuple[str, str]:
    """Takes in a word we want to spell, a list of words to sample from, and the number of shots.
    Tuple is of the form (word, spelling).
    Creates a prompt that asks for the first letter of a word.
    
    Returns the prompt and the answer to the prompt."""
    assert 0 <= num_shots < len(word_list), "Number of shots must be between 0 and the number of words in the list, minus the chosen word."
    prompt = ''
    if word in word_list:
        word_list.remove(word) # Ensure we don't sample the same word we want to spell.
    if num_shots > 0:
        samples = random.sample(word_list, num_shots)
        for sample in samples:
            prompt += f"Q: What is the first letter of '{sample[0]}'? A: {sample[1][0]}\n\n"
    prompt += f"Q: What is the first letter of '{word}'? A:"
    return prompt, word[0]


def format_first_letter_response(item):
    """Format the response to a first letter spelling prompt."""
    return format_full_spelling_response(item)[0]


# What position is the letter in the word eval.
INDEX_TO_POSITION = {0: 'first', 1: 'second', 2: 'third', 3: 'fourth', 4: 'fifth', 5: 'sixth', 6: 'seventh', 
                     7: 'eighth', 8: 'ninth', 9: 'tenth', 10: 'eleventh', 11: 'twelfth'}

def create_position_of_letter_spelling_prompt(word: str, word_list: Tuple[str, str], num_shots: int):
    """Takes in a word we want to spell, a list of words to sample from, and the number of shots.
    Tuple is of the form (word, spelling).
    Creates a prompt that asks what position a letter of a word is in.
    
    Returns the prompt and the answer to the prompt."""
    assert 0 <= num_shots < len(word_list), "Number of shots must be between 0 and the number of words in the list, minus the chosen word."
    prompt = ''
    if word in word_list:
        word_list.remove(word) # Ensure we don't sample the same word we want to spell.
    if num_shots > 0:
        samples = random.sample(word_list, num_shots)
        for sample in samples:
            position = random.randint(0, len(sample[0])-1)
            prompt += f"Q: What is the {INDEX_TO_POSITION[position]} letter of '{sample[0]}'? A: {sample[1][position]}\n\n"
    position = random.randint(0, len(word[0])-1)
    prompt += f"Q: What is the {INDEX_TO_POSITION[position]} letter of '{word}'? A:"
    return prompt, word[position]


def format_position_of_letter_response(item):
    """Format the response to a letter in word spelling prompt."""
    return format_full_spelling_response(item)[0]


def get_position_of_letter_accuracy(data: Dict[int, List[SpellingEvalDict]]) -> Dict[int, float]:
    """Takes in a dictionary of results, and returns the accuracy of the model on each grade."""
    return {grade: sum([1 if d['response'].startswith(d['answer']) else 0 for d in data[grade]]) / len(data[grade]) for grade in data.keys()}


def get_spelling_accuracy(data: Dict[int, List[SpellingEvalDict]]) -> Dict[int, float]:
    """Takes in a dictionary of results, and returns the accuracy of the model on each grade."""
    return {grade: sum([1 if d['response'].startswith(d['answer']) else 0 for d in data[grade]]) / len(data[grade]) for grade in data.keys()}


def run_multiple_shots(shots: List[int], eval: SpellingEval, 
                       model, model_type: ModelType, tokenizer, 
                       word_list: List[Tuple[str, str]],
                       response_type: EvalResponse,
                       batch_size: int=10) -> Dict[int, List[SpellingEvalDict]] | Dict[int, float]:
    """Run an eval for multiple few-shot parameters."""
    shots = {shot: {} for shot in shots}
    for shot in shots:
        shots[shot] = eval.run_eval(model, model_type, tokenizer, response_type, word_list, shot, batch_size)
    return shots