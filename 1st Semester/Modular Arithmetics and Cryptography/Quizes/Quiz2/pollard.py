import math


class PollardSolver:
    _MAX_ITER = 10

    @classmethod
    def decompose(
        cls, n: int, x_0: int, step_func: callable, *, verbose: bool = False
    ) -> (int, int):
        x = [x_0]
        finished = False

        def step():
            x.append(step_func(x[-1]) % n)
            if verbose:
                print(f"x_{len(x) - 1} = {x[-1]}", end = ", ")

        for idx in range(1, cls._MAX_ITER + 1):
            if finished:
                if verbose:
                    print(f"x_{2 * idx - 1} = x, x_{2 * idx} = x", end=', ')
                    print(f"(|x_{2 * idx} - x_{idx}|, {n}) = x")
                    continue
                else:
                    break
            step()
            step()
            gcd = math.gcd(abs(x[2 * idx] - x[idx]), n)
            if verbose:
                print(f"(|x_{2 * idx} - x_{idx}|, n) = {gcd}")
            if gcd != 1:
                finished = True
        
        if not finished:
            raise ValueError(f"Pollard solver failed within {cls._MAX_ITER} iterations.")
        
        return tuple(sorted([gcd, n // gcd]))


if __name__ == "__main__":
    n = int(input("n = "))
    x_0 = 2
    step_func = lambda x: x**2 + 1
    print(f"{x_0 = }, step_func = x**2 + 1")
    factors = PollardSolver.decompose(
        n, x_0, step_func, verbose=True
    )
    print(f"{factors = }")


