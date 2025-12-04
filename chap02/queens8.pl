% permutation/2 (pure)
perm([], []).
perm(L, [H|T]) :- select(H, L, R), perm(R, T).
select(X, [X|T], T).
select(X, [H|T], [H|R]) :- select(X, T, R).

% no two queens attack diagonally
safe([]).
safe([Q|Qs]) :-
    safe(Qs),
    noattack(Q, Qs, 1).

noattack(_, [], _).
noattack(Q, [R|Rs], D) :-
    abs(Q - R) =\= D,
    D1 is D + 1,
    noattack(Q, Rs, D1).

queens8(Qs) :-
    perm([1,2,3,4,5,6,7,8], Qs),
    safe(Qs).
