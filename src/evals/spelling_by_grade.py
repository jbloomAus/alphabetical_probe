from evals.eval_utils import get_spelling, run_inference_on_model, EvalResponse, ModelType
from typing import Dict, List, Literal, Tuple, TypedDict

import random


# Should this be in eval_utils?
class SpellingEvalDict(TypedDict):
    prompt: str
    answer: str
    response: str
    
    
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
    
    def create_prompt(self, word: str, word_list: Tuple[str, str], num_shots: int) -> str:
        """Create a prompt for the model to answer."""
        return self.prompt_function(word, word_list, num_shots)
    
    def format_response(self, response: str) -> str:
        """Format the responses the model gives as needed."""
        return self.format_response_function(response)
    
    def get_accuracy(self, data: Dict[int, List[SpellingEvalDict]]) -> Dict[int, float]:
        """Takes in a dictionary of {grade: [{prompt: str, answer: str, response: str}, ...], ...
        Returns the accuracy of the model on each grade according to the eval's metric function."""
        return self.metric_function(data)
    
    def run_eval(self, model, model_type: ModelType, tokenizer, 
                 return_type: EvalResponse,
                 word_list: List[Tuple[str, str]], 
                 num_shots: int, batch_size: int=10) -> Dict[int, List[SpellingEvalDict]] | Dict[int, float]:
        """Runs the eval on a given word list and number of shots.
        
        args:
        model: Contains the HuggingFace or TransformerLens model to run inference on.
        model_type: Determines what type of model we should use, which determines how we run inference.
        tokenizer: Contains the tokenizer to apply to prompts.
        return_type: Determines what response the eval should return. Data returns the raw data, accuracy returns the accuracy of the model's responses.
        word_list: A list of tuples of the form (word, spelling) to run the eval on.
        num_shots: How many shots to give the model in evaluation.
        batch_size: How many prompts to pass in at once to the model.
        
        Returns:
        If return_type is EvalResponse.DATA, returns an object containing a list of {'prompt': prompt, 'answer': answer, 'response': response} dicts.
        If return_type is EvalResponse.ACCURACY, returns a dictionary of {grade: accuracy} dicts.
        """
        data = {grade: [] for grade in word_list.keys()}
    
        for grade in word_list:
            print(f"Assessing Grade {grade}")
            prompts = [eval.prompt_function(word[0], word_list[grade], num_shots) for word in word_list[grade]]
            data[grade] = run_inference_on_model(model, model_type, tokenizer, prompts, [w[1] for w in word_list[grade]], batch_size)
            data[grade] = [{'prompt': item['prompt'], 
                            'answer': item['answer'], 
                            'response': eval.format_response(item['response'])} 
                        for item in data[grade]]
        return data if return_type == EvalResponse.DATA else eval.get_accuracy(data)
        
    

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


def create_full_spelling_prompt(word: str, word_list: Tuple[str, str], num_shots: int):
    """Takes in a word we want to spell, a list of words to sample from, and the number of shots.
    Tuple is of the form (word, spelling).
    Creates a prompt that asks for the full spelling of a word."""
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


def format_full_spelling_response(item):
    """Format the response to a full spelling prompt."""
    return item['response'].split('A: ')[-1].split("\n\n")[0].replace('\n', '').replace('<|endoftext|>', '').strip()


def create_first_letter_spelling_prompt(word: str, word_list: Tuple[str, str], num_shots: int):
    """Takes in a word we want to spell, a list of words to sample from, and the number of shots.
    Tuple is of the form (word, spelling).
    Creates a prompt that asks for the first letter of a word."""
    assert 0 <= num_shots < len(word_list), "Number of shots must be between 0 and the number of words in the list, minus the chosen word."
    prompt = ''
    if word in word_list:
        word_list.remove(word) # Ensure we don't sample the same word we want to spell.
    if num_shots > 0:
        samples = random.sample(word_list, num_shots)
        for sample in samples:
            prompt += f"Q: What is the first letter of '{sample[0]}'? A: {sample[1][0]}\n\n"
    prompt += f"Q: What is the first letter of '{word}'? A:"
    return prompt


def format_first_letter_response(item):
    """Format the response to a first letter spelling prompt."""
    return format_full_spelling_response(item)[0]


def create_letter_in_word_spelling_prompt(word: str, word_list: Tuple[str, str], num_shots: int):
    """Takes in a word we want to spell, a list of words to sample from, and the number of shots.
    Tuple is of the form (word, spelling).
    Creates a prompt that asks if a letter is contained within a word."""
    assert 0 <= num_shots < len(word_list), "Number of shots must be between 0 and the number of words in the list, minus the chosen word."
    prompt = ''
    if word in word_list:
        word_list.remove(word) # Ensure we don't sample the same word we want to spell.
    if num_shots > 0:
        samples = random.sample(word_list, num_shots)
        for sample in samples:
            prompt += f"Q: What is the first letter of '{sample[0]}'? A: {sample[1][0]}\n\n"
    prompt += f"Q: What is the first letter of '{word}'? A:"
    return prompt


def get_spelling_accuracy(data: Dict[int, List[SpellingEvalDict]]) -> Dict[int, float]:
    """Takes in a dictionary of results, and returns the accuracy of the model on each grade."""
    return {grade: sum([1 if d['response'].startswith(d['answer']) else 0 for d in data[grade]]) / len(data[grade]) for grade in data.keys()}