import nltk

from typing import List
import numpy as np


def recognize(grammar: nltk.grammar.CFG, sentence: List[str]) -> bool:
    """
    Recognize whether a sentence in the language of grammar or not.

    Args:
        grammar: Grammar rule that is used to determine grammaticality of sentence.
        sentence: Input sentence that will be tested.

    Returns:
        truth_value: A bool value to determine whether if the sentence
        is in the grammar provided or not.
    """
    # YOUR CODE HERE
    #     TODO:
    #         1) Implement the CKY algorithm and use it as a recognizer.

    ############################ STUDENT SOLUTION ###########################
    # CKY algorithm from pseudocode in wikipedia
    # Initialize the table
    len_s = len(sentence)
    table = np.zeros((len_s, len_s + 1), dtype=set)

    for j in range(1, len_s + 1):
        table[j - 1, j] = [prod.lhs() for prod in grammar.productions(rhs=sentence[j - 1])]

        for i in range(j - 2, -1, -1):
            list_lhs = []
            for k in range(i + 1, j):

                R = table[i, k]
                D = table[k, j]
                for nt1 in R:  # right
                    for nt2 in D:  # down
                        [list_lhs.append(prod.lhs()) for prod in grammar.productions()
                         if prod.rhs() == (nt1, nt2)]
            table[i, j] = list_lhs

    return True if grammar.start() in table[0, len_s] else False
#########################################################################