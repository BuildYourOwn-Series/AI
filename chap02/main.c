#include <stdio.h>
#include <SWI-Prolog.h>

int main(int argc, char **argv)
{
    // Initialize Prolog engine
    if ( !PL_initialise(argc, argv) )
        PL_halt(1);

    // Load a knowledge base (file "services.pl")
    PL_call(PL_new_atom("consult('services.pl')"), NULL);

    // Example: query ?- unhealthy(S).
    predicate_t pred = PL_predicate("unhealthy", 1, NULL);
    term_t arg = PL_new_term_refs(1);
    qid_t qid = PL_open_query(NULL, PL_Q_NORMAL, pred, arg);

    while ( PL_next_solution(qid) )
    {
        char *atom;
        if ( PL_get_atom_chars(arg, &atom) )
            printf("Service %s is unhealthy\n", atom);
    }

    PL_close_query(qid);
    PL_halt(0);
}
