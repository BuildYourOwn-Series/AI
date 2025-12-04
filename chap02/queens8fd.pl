:- use_module(library(clpfd)).

queens_fd(Qs) :-
    length(Qs, 8),
    Qs ins 1..8,                 % each queen in a column 1..8
    all_different(Qs),           % no shared columns
    numlist(1,8,Is),
    % diagonals: Qi + i all different, and Qi - i all different
    maplist(sum_with_index,  Qs, Is, SumL),
    maplist(diff_with_index, Qs, Is, DiffL),
    all_different(SumL),
    all_different(DiffL),
    labeling([ffc,bisect], Qs).  % search strategy

sum_with_index(Q,I,S)  :- S #= Q + I.
diff_with_index(Q,I,D) :- D #= Q - I.
