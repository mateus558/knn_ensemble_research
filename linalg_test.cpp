#include <stdlib.h>
#include <stdio.h>
#include <math.h>



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

    return 0;
}