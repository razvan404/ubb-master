class RsaEncryption:
    _ALPHABET = [
        '_', *map(chr, range(ord('A'), ord('Z') + 1))
    ]

    def __init__(self, plain_units: int = 2, cipher_units: int = 3):
        self.k = plain_units
        self.l = cipher_units

    @classmethod
    def _find_smallest_odd_prime_with(cls, num: int) -> int:
        previous_primes = [2]
        e = 3

        while True:
            for previous_prime in previous_primes:
                if e % previous_prime == 0:
                    break
            else:
                if num % e != 0:
                    return e
                previous_primes.append(e)
            e += 1

    @classmethod
    def _variables_from_pq(cls, p: int, q: int, *, public: bool = False, verbose: bool = False):
        n = p * q
        phi_n = (p - 1) * (q - 1)
        e = cls._find_smallest_odd_prime_with(phi_n)
        d = pow(e, -1, phi_n)

        if verbose:
            print(f"{n = }\nphi(n) = {phi_n}\n{e = }")
            if not public:
                print(f"{d = }")

        return n, e if public else d

    @classmethod
    def _text_block_to_numerical(cls, text_block: str):
        numerical = 0
        exponent = 1
        for char in reversed(text_block):
            numerical += exponent * cls._ALPHABET.index(char)
            exponent *= len(cls._ALPHABET)
        return numerical

    @classmethod
    def _numerical_block_to_text(cls, size: int):
        def func(numerical_block: int):
            letters = []
            while numerical_block:
                letters.append(cls._ALPHABET[numerical_block % len(cls._ALPHABET)])
                numerical_block //= len(cls._ALPHABET)
            return cls._ALPHABET[0] * (size - len(letters)) + ''.join(reversed(letters))

        return func

    def _text_to_numerical_blocks(self, text: str, *, public: bool = False, verbose: bool = False):
        block_size = self.k if public else self.l
        plain_blocks = [
            text[idx:idx+block_size]
            for idx in range(0, len(text), block_size)
        ]
        if verbose:
            print(
                f"Blocks of {'k' if public else 'l'} letters:",
                ", ".join(plain_blocks),
            )

        num_equivs = list(map(self._text_block_to_numerical, plain_blocks))
        if verbose:
            print("Numerical equivalents:")
            print("\n".join([
                f"{'b' if public else 'c'}{idx} = {equiv}"
                for idx, equiv in enumerate(num_equivs, start=1)
            ]))

        return num_equivs

    def _numerical_blocks_to_text(self, numerical_blocks: list[int], *, public: bool = False, verbose: bool = False):
        block_size = self.l if public else self.k
        block_of_letters = list(
            map(self._numerical_block_to_text(block_size), numerical_blocks)
        )
        if verbose:
            print(f"Block of {'l' if public else 'c'} letters:", ", ".join(block_of_letters))
        return ''.join(block_of_letters)


    def encrypt(self, text: str, p: int, q: int, *, verbose: bool = False):
        if verbose:
            print("\nValues:")
        n, e = self._variables_from_pq(
            p, q, public=True, verbose=verbose
        )

        if verbose:
            print("\nPlaintext:")
        numerical_blocks = self._text_to_numerical_blocks(
            text, public=True, verbose=verbose
        )

        if verbose:
            print("\nEncryption:")
        encripted_blocks = list(map(lambda x: pow(x, e, n), numerical_blocks))
        if verbose:
            print("\n".join([
                f"c_{idx} = b_{idx}^e mod n = {enc_block}"
                for idx, enc_block in enumerate(encripted_blocks, start=1)
            ]))

        ciphertext = self._numerical_blocks_to_text(
            encripted_blocks, public=True, verbose=verbose
        )
        return ciphertext

    def decrypt(self, ciphertext: str, p: int, q: int, *, verbose: bool = False):
        if verbose:
            print("\nValues:")
        n, d = self._variables_from_pq(
            p, q, public=False, verbose=verbose
        )

        if verbose:
            print("\nCiphertext:")
        numerical_blocks = self._text_to_numerical_blocks(
            ciphertext, public=False, verbose=verbose
        )

        if verbose:
            print("\nDecryption:")
        decripted_blocks = list(map(lambda x: pow(x, d, n), numerical_blocks))
        if verbose:
            print("\n".join([
                f"b_{idx} = c_{idx}^d mod n = {enc_block}"
                for idx, enc_block in enumerate(decripted_blocks, start=1)
            ]))

        text = self._numerical_blocks_to_text(
            decripted_blocks, public=False, verbose=verbose
        )
        return text



if __name__ == "__main__":
    print("alphabet =", " ".join(RsaEncryption._ALPHABET))
    k, l = 2, 3
    rsa_encryption = RsaEncryption(plain_units=k, cipher_units=l)
    print(f"{k = }, {l = }")
    task = input("task (e[ncryption] / d[ecryption]) = ")
    if task in ['e', 'encryption']:
        text = input('text = ')
        p, q = int(input('p = ')), int(input('q = '))
        ciphertext = rsa_encryption.encrypt(text, p, q, verbose=True)
        print(f"\n{ciphertext = }")
    elif task in ['d', 'decryption']:
        ciphertext = input('ciphertext = ')
        p, q = int(input('p = ')), int(input('q = '))
        text = rsa_encryption.decrypt(ciphertext, p, q, verbose=True)
        print(f"\n{text = }")
    else:
        raise ValueError(f"Invalid task: {task}")