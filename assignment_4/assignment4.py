import argparse
import nltk

from nltk.tree import Tree
from model.recognizer import recognize
from model.parser import parse, count
from nltk.draw.tree import draw_trees


GRAMMAR_PATH = './data/atis-grammar-cnf.cfg'


def main():
    parser = argparse.ArgumentParser(
        description='CKY algorithm'
    )

    parser.add_argument(
        '--structural', dest='structural',
        help='Derive sentence with structural ambiguity',
        action='store_true'
    )

    parser.add_argument(
        '--recognizer', dest='recognizer',
        help='Execute CKY for word recognition',
        action='store_true'
    )

    parser.add_argument(
        '--parser', dest='parser',
        help='Execute CKY for parsing',
        action='store_true'
    )

    parser.add_argument(
        '--count', dest='count',
        help='Compute number of parse trees from chart without \
              actually computing the trees (Extra Credit)',
        action='store_true'
    )

    args = parser.parse_args()

    # load the grammar
    grammar = nltk.data.load(GRAMMAR_PATH)
    # load the raw sentences
    s = nltk.data.load("grammars/large_grammars/atis_sentences.txt", "auto")
    # extract the test sentences
    t = nltk.parse.util.extract_test_sentences(s)

    if args.structural:
        # YOUR CODE HERE
        #     TODO:
        #         1) Like asked in the instruction, derive at least two sentences that
        #         exhibit structural ambiguity and indicate the different analyses
        #         (at least two per sentence) with a syntactic tree.
        draw_trees(Tree.fromstring('(S(NP(N(I)))(VP(VP(V(watch))(NP(D(the))(NP(man))))(PP(P(with))(NP(D(the))(N(telescope))))))'),
                   Tree.fromstring('(S (NP (N I)) (VP (V watch) (NP (NP (D the) (N man)) (PP (P with) (NP (D(the))(N (telescope))) ))))'),
                   Tree.fromstring('(S (NP (N (He)) ) (VP (VP (V (works)) (P (out))) (PP(P(in)) (NP (D(the))(N(backyard))))))'),
                   Tree.fromstring('(S (NP (N (He)) ) (VP (VP (V (works)) ) (PP(PP(PP(P(out))(P(in))) (NP (D(the))(N(backyard)))) ) ))'))
    elif args.recognizer:
        # YOUR CODE HERE
        #     TODO:
        #         1) Provide a list of grammatical and ungrammatical sentences (at least 10 each)
        #         and test your recognizer on these sentences.

        grammatical = [t[21][0], t[22][0], t[24][0], t[27][0], t[0][0], t[1][0], t[2][0], t[3][0], t[15][0], t[19][0]]
        ungrammatical = [t[4][0]], [t[60][0]], [t[6][0]], [t[17][0]], [t[26][0]], [t[28][0]], [t[9][0]], [t[13][0]], [t[16][0]], [t[10][0]]
        
        for sents in grammatical:
            val = recognize(grammar, sents)
            if val:
                print("{} is in the language of CFG.".format(sents))
            else:
                print("{} is not in the language of CFG.".format(sents))

        for sents in ungrammatical:
            val = recognize(grammar, sents)
            if val:
                print("{} is in the language of CFG.".format(sents))
            else:
                print("{} is not in the language of CFG.".format(sents))

    elif args.parser:
        # We test the parser by using ATIS test sentences.
        print("ID\t Predicted_Tree\tLabeled_Tree")
        for idx, sents in enumerate(t):
            tree = parse(grammar, sents[0])
            print("{}\t {}\t \t{}".format(idx, len(tree), sents[1]))

        # YOUR CODE HERE
        #     TODO:
        #         1) Choose an ATIS test sentence with a number of parses p
        #         such that 1 < p < 5. Visualize its parses. You can use `draw` 
        #         method to do this.

    elif args.count:
        print("ID\t Predicted_Tree\tLabeled_Tree")
        for idx, sents in enumerate(t):
            num_tree = count(grammar, sents[0])
            print("{}\t {}\t \t{}".format(idx, num_tree, sents[1]))


if __name__ == "__main__":
    main()
