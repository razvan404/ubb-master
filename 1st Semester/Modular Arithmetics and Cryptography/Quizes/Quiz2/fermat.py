import math


class FermatSolver:
    _MAX_ITER = 20

    @classmethod
    def decompose(cls, n: int, *, verbose: bool = False) -> (int, int):
        t_0 = int(math.sqrt(n))
        finished = False
        
        if verbose:
            print(f"t_0 = int(sqrt(n)) = {t_0}")
        
        for idx in range(1, cls._MAX_ITER + 1):
            if finished:
                if verbose:
                    print(f"t = t_0 + {idx}: t**2 - n = x, perfect square (yes / no): x")
                    continue
                else:
                    break

            t = t_0 + idx
            s_2 = t ** 2 - n
            
            if verbose:
                print(f"t = t_0 + {idx}: t**2 - n = {s_2}", end=', ')

            s = int(math.sqrt(s_2))
            if s_2 == s * s:
                finished = True

            if verbose:
                print(f"perfect square (yes / no): {'yes' if finished else 'no'}")

        if not finished:
            raise ValueError(f"Fermat solver failed within {cls._MAX_ITER} iterations.")

        if verbose:
            print(f"{s = }, {t = }")
        return (t - s, t + s)


if __name__ == "__main__":
    n = int(input("n = "))
    factors = FermatSolver.decompose(n, verbose=True)
    print(f"{factors = }")
