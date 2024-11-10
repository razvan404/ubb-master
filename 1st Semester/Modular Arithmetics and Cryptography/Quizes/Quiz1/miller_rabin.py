class MillerRabinSolver:
    _MAX_BITS: int = 10
    _MAX_S: int = 4

    @classmethod
    def _decompose(cls, n: int, *, verbose: bool) -> (int, int):
        s = 0
        t = n - 1
        while t % 2 == 0:
            s += 1
            t //= 2
        if verbose:
            print(f"{s = }, {t = }")
        if s > cls._MAX_S:
            raise ValueError("Max s ({cls._MAX_S}) exceeded.")
        return s, t

    @classmethod
    def _to_bits(cls, t: int, *, verbose: bool) -> list[int]:
        t_bits = []
        while t > 0:
            t_bits.append(t % 2)
            t //= 2
            if len(t_bits) > cls._MAX_BITS:
                raise ValueError(f"Max bits ({cls._MAX_BITS}) exceeded.")
        t_bits = list(reversed(t_bits))
        if verbose:
            print(f"t in binary = {'0' * (cls._MAX_BITS - len(t_bits))}{''.join(map(str, t_bits))}")
        return t_bits

    @classmethod
    def _factor_powers_mod_n(cls, n: int, factor: int, num_bits: int, *, verbose: bool) -> list[int]:
        factor_powers = [factor]
        if verbose:
            print(f"{factor}^(2^0) = {factor}")
        for k in range(1, cls._MAX_BITS):
            if k >= num_bits:
                if verbose:
                    print(f"{factor}^(2^{k}) = x")
                    continue
                else:
                    break
            factor_powers.append((factor_powers[-1] ** 2) % n)
            if verbose:
                print(f"{factor}^(2^{k}) = {factor_powers[-1]}")
        return factor_powers

    @classmethod
    def _compute_factor_pow_t_mod_n(
        cls, factor_powers: list[int], t_bits: list[int], n: int, *, verbose: bool
    ):
        factor_pow_t = 1
        for factor_power, bit in zip(factor_powers, reversed(t_bits)):
            if bit == 1:
                factor_pow_t = (factor_pow_t * factor_power) % n
        if verbose:
            print(f"{factor_powers[0]}^t = {factor_pow_t}")
        return factor_pow_t

    @classmethod
    def _compute_sequence(cls, factor: int, factor_pow_t: int, s: int, *, verbose: bool) -> list[int]:
        sequence = [factor_pow_t]
        for i in range(1, max(s + 1, cls._MAX_S + 1)):
            if i > s:
                print(f"{factor}^(2^{i}*t) = x")
                continue
            sequence.append((sequence[-1] ** 2) % n)
            if verbose:
                print(f"{factor}^(2^{i}*t) = {sequence[-1]}")
        return sequence

    @classmethod
    def test_primality(
        cls, n: int, *, verbose: bool = False, iterations_factors: list[int]
    ) -> bool:
        s, t = cls._decompose(n, verbose=verbose)
        t_bits = cls._to_bits(t, verbose=verbose)
        is_composite = False
        for k, a in enumerate(iterations_factors, start=1):
            if verbose:
                print("=" * 20)
                print(f"Iteration {k = }, {a = }:\n")

            if not is_composite:
                should_print_factor_powers = verbose and k == 1
                factor_powers = cls._factor_powers_mod_n(n, a, len(t_bits), verbose=should_print_factor_powers)
                if should_print_factor_powers:
                    print()
                a_pow_t = cls._compute_factor_pow_t_mod_n(factor_powers, t_bits, n, verbose=verbose)
                sequence = cls._compute_sequence(a, a_pow_t, s, verbose=verbose)

                try:
                    index_1 = sequence.index(1)
                except:
                    index_1 = None

                if index_1 is None or not (
                    index_1 == 0 or sequence[index_1 - 1] == n - 1
                ):
                    is_composite = True
            elif verbose:
                print(f"{a}^t = x")
                for i in range(1, cls._MAX_S + 1):
                    print(f"{a}^(2^{i}*t) = x")

            if verbose:
                    print("=" * 20)
        return not is_composite


if __name__ == "__main__":
    n = int(input("n = "))
    is_prime = MillerRabinSolver.test_primality(n, verbose=True, iterations_factors=[2, 3, 5])
    print(f"n is prime: {'yes' if is_prime else 'no'}")