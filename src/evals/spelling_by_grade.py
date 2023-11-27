from evals.eval_utils import get_spelling, run_inference_on_model, ModelType
from typing import Callable, Dict, List, Tuple, TypedDict

import random
import string

class SpellingEvalDict(TypedDict):
    """Defines the structure of a dictionary containing the data for a spelling eval prompt.
    Contains the word, prompt, answer, raw response, and formatted response."""
    word: str
    prompt: str
    answer: str
    response: str
    formatted_response: str
    

class SpellingEvalResponseDict(TypedDict):
    """Defines the structure of a dictionary containing the data and accuracy for a spelling eval."""
    data: Dict[int, List[SpellingEvalDict]]
    accuracy: Dict[int, float]
    
SpellingData = Dict[int, List[SpellingEvalDict]] # Dictionary of spelling eval data by grade.
    
# word=str, word_list = List[Tuple[str, str]], num_shots=int. Returns Tuple[str, str] as (prompt, answer).
PromptFunctionType = Callable[[str, List[Tuple[str, str]], int], Tuple[str, str]]  

class GradeSpellingEval:
    """An object that contains the functions needed to run an eval on spelling by grade.
    Together, the functions define the evaluation.
    
    args:
    prompt_function: A function that takes in a word, a list of words to sample from, and the number of shots.
    Returns a (prompt, answer) pair.
    format_response_function: A function that takes in the response from the model as a string and formats it as needed.
    Returns another string.
    metric_function: A function that takes in a dictionary of {grade: [{prompt: str, answer: str, response: str}, ...], ...
    and judges the accuracy of the responses based on the answers.
    Returns a dictionary of {grade: accuracy}, with accuracy between 0 and 1.
    filter_function: A function that takes in a dataset item as a string and returns True 
    if it should be included in the eval. Returns a boolean."""
    
    def __init__(self, name, prompt_function: PromptFunctionType,
                 format_response_function: Callable[[str], str] = lambda x: x,
                 metric_function: Callable[[SpellingData], Dict[int, float]] = lambda x: 1 if x['answer'] == x['formatted_response'] else 0,
                 filter_function: Callable[[str], bool] = lambda x: True):
        self.name = name
        self.prompt_function = prompt_function
        self.format_response_function = format_response_function
        self.metric_function = metric_function
        self.filter_function = filter_function
    
    def run_eval(self, model, model_type: ModelType, tokenizer, 
                 word_list: Dict[int, List[Tuple[str, str]]], 
                 num_shots: int, batch_size: int=10) -> SpellingEvalResponseDict:
        """Runs the eval on a given word list and number of shots.
        
        For each grade in the word list:
        - Filter the dataset according to the filter function (default: Return True for everything)
        - For each batch of words:
            - Create a prompt, answer pair for each word in the batch according to the prompt function.
            - Run these on the selected model.
            - Format the responses according to the format_response_function (default: return the response as-is)
        - Return the raw data and the accuracy metrics according to the metric_function (default: 1 if answer == response, else 0)
        
        args:
        model: Contains the HuggingFace or TransformerLens model to run inference on.
        model_type: Determines what type of model we should use, which determines how we run inference.
        tokenizer: Contains the tokenizer to apply to prompts.
        word_list: A list of tuples of the form (word, spelling) to run the eval on, separated by grade.
        num_shots: How many shots to give the model in evaluation.
        batch_size: How many prompts to pass in at once to the model.
        
        Returns:
        A SpellingEvalResponseDict containing the data and accuracy of the eval.
        """
        data = {grade: [] for grade in word_list.keys()}
    
        for grade in word_list:
            print(f"Assessing Grade {grade}")
            word_list[grade] = [item for item in word_list[grade] if self.filter_function(item)]
            prompts, answers = zip(*[self.prompt_function(word[0], word_list[grade], num_shots) for word in word_list[grade]])
            data[grade] = run_inference_on_model(model, model_type, tokenizer, prompts, answers, batch_size)
            for item in data[grade]:
                item['formatted_response'] = self.format_response_function(item['response'])
 
        response: SpellingEvalResponseDict = {'data': data, 'accuracy': self.metric_function(data)}
        return response
    
    def run_eval_with_multiple_shots(self, model, model_type: ModelType, tokenizer, 
                                     word_list: Dict[int, List[Tuple[str, str]]], 
                                     shots: List[int], batch_size: int=10) -> Dict[int, SpellingEvalResponseDict]:
        """Runs the eval on a given word list for multiple few-shot parameters.
        Same as run_eval but for multiple few-shot parameters.
        
        Returns: A dictionary of {num_shots: EvalResponseDict}."""
        shots = {shot: {} for shot in shots}
        for shot in shots:
            shots[shot] = self.run_eval(model, model_type, tokenizer, word_list, shot, batch_size)
        return shots    
        
        
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
                    if lines[l].strip() != '':
                        words_by_grade[i+1].append((lines[l].strip(), get_spelling(lines[l].strip(), separator, case)))

    return words_by_grade

# Full spelling eval
def create_full_spelling_prompt(word: str, word_list: List[Tuple[str, str]], num_shots: int) -> Tuple[str, str]:
    """Takes in a word we want to spell, a list of words to sample from, and the number of shots.
    Tuples are of the form (word, spelling).
    Creates a prompt that asks for the full spelling of a word.
    
    Returns the prompt and the answer to the prompt."""
    assert 0 <= num_shots < len(word_list), "Number of shots must be between 0 and the number of words in the list, minus the chosen word."
    _, answer = [w for w in word_list if w[0] == word][0] # Assumes unique words in word_list. TODO: Improve?
    word_list = [item for item in word_list if item[0] != word] # Remove any words that are the same as the word we want to spell.
    prompt = ''
    if num_shots > 0:
        samples = random.sample(word_list, num_shots)
        for sample in samples:
            prompt += f"Q: How do you spell '{sample[0]}'? A: {sample[1]}\n\n"
    prompt += f"Q: How do you spell '{word}'? A:"
    return prompt, answer


def format_full_spelling_response(item: str) -> str:
    """Format the response to a full spelling prompt."""
    return item.split('A: ')[-1].split("\n\n")[0].replace('\n', '').replace('<|endoftext|>', '').strip()


def get_spelling_accuracy(data: Dict[int, List[SpellingEvalDict]]) -> Dict[int, float]:
    """Takes in a dictionary of results, and returns the accuracy of the model on each grade."""
    return {grade: sum([1 if d['answer'] in d['formatted_response'].upper() else 0 for d in data[grade]]) / len(data[grade]) for grade in data.keys()}


# First letter eval
def create_first_letter_spelling_prompt(word: str, word_list: List[Tuple[str, str]], num_shots: int) -> Tuple[str, str]:
    """Takes in a word we want to spell, a list of words to sample from, and the number of shots.
    Tuple is of the form (word, spelling).
    Creates a prompt that asks for the first letter of a word.
    
    Returns the prompt and the answer to the prompt."""
    assert 0 <= num_shots < len(word_list), "Number of shots must be between 0 and the number of words in the list, minus the chosen word."
    word_list = [item for item in word_list if item[0] != word] # Remove any words that are the same as the word we want to spell.
    prompt = ''
    if num_shots > 0:
        samples = random.sample(word_list, num_shots)
        for sample in samples:
            prompt += f"Q: What is the first letter of '{sample[0]}'? A: {sample[1][0].upper()}\n\n"
    prompt += f"Q: What is the first letter of '{word}'? A:"
    return prompt, word[0].upper()


def format_first_letter_response(item: str) -> str:
    """Format the response to a first letter spelling prompt."""
    return format_full_spelling_response(item)[0]


# What position is the letter in the word eval.
INDEX_TO_POSITION = {1: 'second', 2: 'third', 3: 'fourth', 4: 'fifth', 5: 'sixth', 6: 'seventh', 
                     7: 'eighth', 8: 'ninth', 9: 'tenth', 10: 'eleventh', 11: 'twelfth'}

def create_position_of_letter_spelling_prompt(word: str, word_list: List[Tuple[str, str]], num_shots: int):
    """Takes in a word we want to spell, a list of words to sample from, and the number of shots.
    Tuple is of the form (word, spelling).
    Creates a prompt that asks what position a letter of a word is in. Does not use the first letter.
    
    Returns the prompt and the answer to the prompt."""
    assert 0 <= num_shots < len(word_list), "Number of shots must be between 0 and the number of words in the list, minus the chosen word."
    word_list = [item for item in word_list if item[0] != word] # Remove any words that are the same as the word we want to spell.
    word_list = [item for item in word_list if len(item[0]) > 1] # Remove any words that are too short.
    prompt = ''
    if num_shots > 0:
        samples = random.sample(word_list, num_shots)
        for sample in samples:
            position = random.randint(1, len(sample[0])-1)
            prompt += f"Q: What is the {INDEX_TO_POSITION[position]} letter of '{sample[0]}'? A: {sample[0][position].upper()}\n\n"
    position = 1 if len(word) == 2 else random.randint(1, len(word)-1)
    prompt += f"Q: What is the {INDEX_TO_POSITION[position]} letter of '{word}'? A:"
    return prompt, word[position].upper()


def format_position_of_letter_response(item: str) -> str:
    """Format the response to a letter in word spelling prompt."""
    return format_full_spelling_response(item)[0]


def get_position_of_letter_accuracy(data: Dict[int, List[SpellingEvalDict]]) -> Dict[int, float]:
    """Takes in a dictionary of results, and returns the accuracy of the model on each grade."""
    return {grade: sum([1 if d['formatted_response'].startswith(d['answer']) else 0 for d in data[grade]]) / len(data[grade]) for grade in data.keys()}

# Is letter in word eval
def create_is_letter_in_word_prompt(word: str, word_list: List[Tuple[str, str]], num_shots: int):
    """Takes in a word we want to spell, a list of words to sample from, and the number of shots.
    Tuple is of the form (word, spelling).
    Creates a prompt that asks if a letter is in a word.
    
    Returns the prompt and the answer to the prompt."""
    assert 0 <= num_shots < len(word_list), "Number of shots must be between 0 and the number of words in the list, minus the chosen word."
    word_list = [item for item in word_list if item[0] != word] # Remove any words that are the same as the word we want to spell.
    word_list = [item for item in word_list if len(item[0]) > 1] # Remove any words that are too short.
    prompt = ''
    if num_shots > 0:
        samples = random.sample(word_list, num_shots)
        for sample in samples:
            position = random.randint(1, len(sample[0])-1)
            
            # We want to pick letters in the word about half the time.
            if random.random() < 0.5:
                prompt += f"Q: Is the letter '{sample[0][position].lower()}' in '{sample[0]}'? A: Yes\n\n"
            else:
                # Pick a random letter that isn't in the word.
                letter = random.choice([char for char in string.ascii_lowercase if char not in sample[0]])
                prompt += f"Q: Is the letter '{letter}' in '{sample[0]}'? A: No\n\n"
    position = 1 if len(word) == 2 else random.randint(1, len(word)-1)
    
    if random.random() < 0.5:
       letter = word[position].lower()
    else:
        # Pick a random letter that isn't in the word.
        letter = random.choice([char for char in string.ascii_lowercase if char not in word])
    prompt += f"Q: Is the letter '{letter}' in '{word}'? A:"
    return prompt, 'Yes' if letter in word else 'No'

def format_is_letter_in_word_response(item: str) -> str:
    """Format the response to a is letter in word spelling prompt."""
    return format_full_spelling_response(item).split(' ')[-1].upper()

def get_is_letter_in_word_accuracy(data: Dict[int, List[SpellingEvalDict]]) -> Dict[int, float]:
    """Takes in a dictionary of results, and returns the accuracy of the model on each grade."""
    return {grade: sum([1 if d['answer'].upper() == d['formatted_response'] else 0 for d in data[grade]]) / len(data[grade]) for grade in data.keys()}

# Scramble word eval
def create_scrambled_spelling_prompt(word: str, word_list: List[Tuple[str, str]], num_shots: int) -> Tuple[str, str]:
    """Takes in a word we want to spell, a list of words to sample from, and the number of shots.
    Tuples are of the form (word, spelling).
    Creates a prompt that asks for the full spelling of a word after slightly scrambling it.
    
    Returns the prompt and the answer to the prompt."""
    assert 0 <= num_shots < len(word_list), "Number of shots must be between 0 and the number of words in the list, minus the chosen word."
    _, answer = [w for w in word_list if w[0] == word][0] # Assumes unique words in word_list. TODO: Improve?
    word_list = [item for item in word_list if item[0] != word] # Remove any words that are the same as the word we want to spell.
    word_list = [item for item in word_list if len(item[0]) > 2] # Remove any words that are too short.
    prompt = ''
    if num_shots > 0:
        samples = random.sample(word_list, num_shots)
        for sample in samples:
            prompt += f"Q: How do you spell '{scramble_word(sample[0])}' correctly? A: {sample[1]}\n\n"
    prompt += f"Q: How do you spell '{scramble_word(word)}' correctly? A:"
    return prompt, answer

def scramble_word(word: str) -> str:
    """Scramble a word by shuffling two of the letters."""
    position = random.randint(1, len(word)-1)
    return word[:position-1] + word[position] + word[position-1] + word[position+1:]