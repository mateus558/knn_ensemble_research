#include "alglib/stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "alglib/optimization.h"

using namespace alglib;


int main(int argc, char **argv)
{
    //
    // This example demonstrates minimization of F(x0,x1) = x0^2 + x1^2 -6*x0 - 4*x1
    // subject to linear constraint x0+x1<=2
    //
    // Exact solution is [x0,x1] = [1.5,0.5]
    //
    // IMPORTANT: this solver minimizes  following  function:
    //     f(x) = 0.5*x'*A*x + b'*x.
    // Note that quadratic term has 0.5 before it. So if you want to minimize
    // quadratic function, you should rewrite it in such way that quadratic term
    // is multiplied by 0.5 too.
    // For example, our function is f(x)=x0^2+x1^2+..., but we rewrite it as
    //     f(x) = 0.5*(2*x0^2+2*x1^2) + ....
    // and pass diag(2,2) as quadratic term - NOT diag(1,1)!
    //
    real_2d_array a = "[[2,0],[0,2]]";
    real_1d_array b = "[-6,-4]";
    real_1d_array s = "[1,1]";
    real_2d_array c = "[[1.0,1.0,2.0]]";
    integer_1d_array ct = "[-1]";
    real_1d_array x;
    minqpstate state;
    minqpreport rep;

    // create solver, set quadratic/linear terms
    minqpcreate(2, state);
    minqpsetquadraticterm(state, a);
    minqpsetlinearterm(state, b);
    minqpsetlc(state, c, ct);

    // Set scale of the parameters.
    // It is strongly recommended that you set scale of your variables.
    // Knowing their scales is essential for evaluation of stopping criteria
    // and for preconditioning of the algorithm steps.
    // You can find more information on scaling at http://www.alglib.net/optimization/scaling.php
    //
    // NOTE: for convex problems you may try using minqpsetscaleautodiag()
    //       which automatically determines variable scales.
    minqpsetscale(state, s);

    //
    // Solve problem with BLEIC-based QP solver.
    //
    // This solver is intended for problems with moderate (up to 50) number
    // of general linear constraints and unlimited number of box constraints.
    //
    // Default stopping criteria are used.
    //
    minqpsetalgobleic(state, 0.0, 0.0, 0.0, 0);
    minqpoptimize(state);
    minqpresults(state, x, rep);
    printf("%s\n", x.tostring(1).c_str()); // EXPECTED: [1.500,0.500]

    //
    // Solve problem with DENSE-AUL solver.
    //
    // This solver is optimized for problems with up to several thousands of
    // variables and large amount of general linear constraints. Problems with
    // less than 50 general linear constraints can be efficiently solved with
    // BLEIC, problems with box-only constraints can be solved with QuickQP.
    // However, DENSE-AUL will work in any (including unconstrained) case.
    //
    // Default stopping criteria are used.
    //
    minqpsetalgodenseaul(state, 1.0e-9, 1.0e+4, 5);
    minqpoptimize(state);
    minqpresults(state, x, rep);
    printf("%s\n", x.tostring(1).c_str()); // EXPECTED: [1.500,0.500]

    //
    // Solve problem with QuickQP solver.
    //
    // This solver is intended for medium and large-scale problems with box
    // constraints, and...
    //
    // ...Oops! It does not support general linear constraints, -5 returned as completion code!
    //
    minqpsetalgoquickqp(state, 0.0, 0.0, 0.0, 0, true);
    minqpoptimize(state);
    minqpresults(state, x, rep);
    printf("%d\n", int(rep.terminationtype)); // EXPECTED: -5
    return 0;
}