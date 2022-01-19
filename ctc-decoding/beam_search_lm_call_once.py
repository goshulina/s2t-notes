# Modified from https://github.com/corticph/prefix-beam-search/blob/master/prefix_beam_search.py
# Modification: call Language Model once per all beams, but not on every beam

from collections import defaultdict, Counter
import re
import numpy as np


def top_n_max(series, top=2, diraction=max):
    series = list(set(series))
    if diraction == max:
      top_result = sorted(series)[-top:]
    else:
      top_result = sorted(series)[:top]
    return top_result


def lang_model_(x, y):
    pass


def prefix_beam_search(ctc,
                       alphabet,
                       lm=None,
                       k=250,
                       alpha=0.5,
                       beta=0.5,
                       prune=0.001
                       ):
    lm = (lambda l: 1) if lm is None else lm
    W = lambda l: re.findall(r'\w+[\s|>]', l)
    F = ctc.shape[1]
    ctc = np.vstack((np.zeros(F), ctc))
    T = ctc.shape[0]
    O = ''
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][O] = 1
    Pnb[0][O] = 0
    A_prev = [O]
    # ks = [1000000 if i < 5 else 10000 for i in range(T)]
    space_flag = False
    for t in range(1, T):
        pruned_alphabet = [alphabet[i] for i in np.where(ctc[t] > prune)[0]]
        top = top_n_max(ctc[t], top=5)
        pruned_alphabet_1 = [alphabet[list(ctc[t]).index(i)] for i in top]
        pruned_alphabet.extend(pruned_alphabet_1)
        pruned_alphabet = list(set(pruned_alphabet))
        if ' ' in pruned_alphabet:
            pruned_alphabet.remove(' ')
            space_flag = True
        for n, l in enumerate(A_prev):
            for c in pruned_alphabet:
                c_ix = alphabet.index(c)
                if c == '%':
                    Pb[t][l] += ctc[t][-1] * (Pb[t - 1][l] + Pnb[t - 1][l])  # lm_prob *
                else:
                    l_plus = l + c
                    if len(l) > 0 and c == l[-1]:
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l]
                        Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]
                    else:
                        Pnb[t][l_plus] += ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    if l_plus not in A_prev:
                        Pb[t][l_plus] += ctc[t][-1] * (Pb[t - 1][l_plus] + Pnb[t - 1][
                            l_plus])
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pnb[t - 1][l_plus]
        if space_flag:
            l_pluses = []
            for l in A_prev:
                try:
                    if len(A_prev) == 1:
                        l_pluses.append(l + ' ')
                    else:
                        if l[-1] != ' ':
                            l_pluses.append(l + ' ')
                except IndexError:
                    pass
            else:
                c_ix = alphabet.index(' ')
                space_flag = False
                lm_prob = lang_model_(l_pluses, lm)
                for i in range(len(l_pluses)):
                    try:
                        if l_pluses[i][-1] == l_pluses[i][-2] == ' ':
                            continue
                    except IndexError:
                        pass
                    Pnb[t][l_pluses[i]] += ctc[t][c_ix] * lm_prob[i] * (Pb[t - 1][A_prev[i]] + Pnb[t - 1][A_prev[i]])
                for i in range(len(l_pluses)):
                    try:
                        if l_pluses[i][-1] == l_pluses[i][-2] == ' ':
                            continue
                    except IndexError:
                        pass
                    if l_pluses[i] not in A_prev:
                        Pb[t][l_pluses[i]] += ctc[t][-1] * (Pb[t - 1][l_pluses[i]] + Pnb[t - 1][l_pluses[i]])
                        Pnb[t][l_pluses[i]] += ctc[t][c_ix] * lm_prob[i] ** alpha * Pnb[t - 1][l_pluses[i]]
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] * (len(W(l)) + 1)  # ** beta
        A_prev = sorted(A_next, key=sorter, reverse=True)[:k]
    return A_prev[0].strip()
