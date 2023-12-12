from evals.spelling_by_grade import *

# Evals
FULL_SPELLING_EVAL = GradeSpellingEval("Spell Full Word", 
                                       create_full_spelling_prompt, 
                                       format_full_spelling_response, 
                                       get_spelling_accuracy)
FULL_SPELLING_PYTHIA_EVAL = GradeSpellingEval("Spell Full Word Pythia",
                                              create_full_spelling_pythia_prompt,
                                              format_full_spelling_response,
                                              get_spelling_accuracy)
GET_FIRST_LETTER_EVAL = GradeSpellingEval("Spell First Letter", 
                                          create_first_letter_spelling_prompt, 
                                          format_first_letter_response, 
                                          get_first_letter_accuracy)
GET_POSITION_OF_LETTER_EVAL = GradeSpellingEval("Get Position of Letter In Word", 
                                                create_position_of_letter_spelling_prompt, 
                                                format_position_of_letter_response, 
                                                get_position_of_letter_accuracy)
IS_LETTER_IN_WORD_EVAL = GradeSpellingEval("Is Letter in Word", 
                                           create_is_letter_in_word_prompt, 
                                           format_is_letter_in_word_response, 
                                           get_is_letter_in_word_accuracy)
SCRAMBLED_SPELLING_EVAL = GradeSpellingEval("Spell Scrambled Word", 
                                            create_scrambled_spelling_prompt, 
                                            format_full_spelling_response, 
                                            get_spelling_accuracy)
GET_NEXT_LETTER_EVAL = GradeSpellingEval("Get Next Letter",
                                         create_next_letter_spelling_prompt,
                                         format_next_letter_response,
                                         get_first_letter_accuracy)

EVAL_LIST = [FULL_SPELLING_EVAL, GET_FIRST_LETTER_EVAL, GET_POSITION_OF_LETTER_EVAL, 
             IS_LETTER_IN_WORD_EVAL, SCRAMBLED_SPELLING_EVAL, GET_NEXT_LETTER_EVAL]