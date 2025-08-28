#!/usr/bin/env python3
"""
Python version of prediction_shadow.cpp with more Pythonic code
and less mapping with the C++ implementation on variable names and structures.

Created by converting from Hsin-Yuan Huang's C++ implementation.
For more details, see the accompanying paper:
"Predicting Many Properties of a Quantum System from Very Few Measurements".

This version is converted by LLM agent,
Github Copilot with the model Claude Sonnet 4,
and Pythonicize manually.

"""

from typing import Literal, Sequence
import sys
import math
from pathlib import Path
import numpy as np


def read_all_subsystems(filename: str | Path) -> tuple[list[list[int]], int]:
    """Read subsystems.txt and return subsystem information.

    Args:
        filename (str | Path): The path to the subsystems file.

    Returns:
        tuple[list[list[int]], int]: A tuple containing a list of subsystems and the system size.
    """

    with open(filename, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()  # This will iterate first line.
        if not first_line:
            raise ValueError("The input file is empty.")

        system_size = int(first_line)

        subsystems = []

        for line in f:
            parts = line.strip().split()
            k_local = int(parts[0])
            subsystems.append([int(parts[k + 1]) for k in range(k_local)])

        return subsystems, system_size


def read_all_measurements(
    measurement_file_name: str | Path,
) -> tuple[list[list[int]], list[list[int]], int]:
    """Read measurement.txt and return measurement data.

    Args:
        filename (str | Path): The path to the measurement file.

    Returns:
        tuple[list[list[int]], list[list[int]], int]:
            A tuple containing a list of pauli basis, a list of spin outcomes, and the system size.
    """
    pauli_basis = []
    spin_outcome = []

    with open(measurement_file_name, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()  # This will iterate first line.
        if not first_line:
            raise ValueError("The input file is empty.")

        system_size = int(first_line)

        for line in f:
            parts = line.strip().split()

            pauli_basis.append(
                [ord(pauli[0]) - ord("X") for pauli in parts[0::2]]
            )  # (X=0, Y=1, Z=2)
            spin_outcome.append([int(outcome) for outcome in parts[1::2]])

    return pauli_basis, spin_outcome, system_size


def count_trailing_zeros(n: int) -> int:
    """Calculate the number of trailing zeros in the binary representation of n.
    Equal to __builtin_ctzll(n) in C++.

    Args:
        n (int): The integer to count trailing zeros in.

    Returns:
        int: The number of trailing zeros in the binary representation of n.
    """
    if n == 0:
        return 64
    count = 0
    while (n & 1) == 0:
        n >>= 1
        count += 1
    return count


def c_non_id(c: int, subsystem_size: int) -> int:
    """Count the number of non-identity Pauli operators in the encoding.

    Description of the process:

    .. code-block:: text

        Gray Code: 0: 00, 1: 01, 2: 11, 3: 10
        Pauli: I: 0, X: 1, Y: 2, Z: 3
        => I: 00, X: 01, Y: 11, Z: 10

        Consider n = 8, max_encoding = 2 ** (2 * subsystem_size)
        For example, bitstring = 0b 00000000,
        and Pauli basis encoding c = 0b 0000 0001 1000 1100

            0000 0001 1000 1100 = II IX ZI YI
        &   1111 1111 1111 1111
        --------------------------------------
            0000 0001 1000 1100 = II IX ZI YI
        For only identity I: 00 keep 00 after AND with 11

        -> II IX ZI YI
        is not identity (I)
        -> 00 01 10 10
        sum(is not identity (I))
        -> non_id = 3

    Original implementaion in Python by GitHub Copliot Claude Sonnet 4:

    .. code-block:: python

        non_id = 0
        for i in range(subsystem_size):
            if ((c >> (2 * i)) & 3) != 0:  # 0b11 = 3
                non_id += 1

    Original implementaion in C++:

    .. code-block:: c++

        int nonId = 0;
        for(int i = 0; i < subsystem_size; i++)
            nonId += ((c >> (2 * i)) & 3) != 0;

    Args:
        c (int): The encoding to check.
        subsystem_size (int): The size of the subsystem.

    Returns:
        int: The number of non-identity Pauli operators.
    """

    return sum(((c >> (2 * i)) & 3) != 0 for i in range(subsystem_size))


def calculate_level_count(
    max_encoding: int, subsystem_size: int, all_number_of_outcomes: Sequence[int]
) -> tuple[list[int], list[int]]:
    """Calculate the level count for a given encoding.

    Original implementaion in Python by GitHub Copliot Claude Sonnet 4:

    .. code-block:: python

        level_cnt = [0] * (subsystem_size + 1)
        level_ttl = [0] * (subsystem_size + 1)

        for c in range(max_encoding):
            non_id = 0
            for i in range(subsystem_size):
                if ((c >> (2 * i)) & 3) != 0:
                    non_id += 1

            if renyi_number_of_outcomes[c] >= 2:
                level_cnt[non_id] += 1
            level_ttl[non_id] += 1

    Original implementaion in C++:

    .. code-block:: c++

        int level_cnt[2 * subsystem_size], level_ttl[2 * subsystem_size];
        for(int i = 0; i < subsystem_size + 1; i++){
            level_cnt[i] = 0;
            level_ttl[i] = 0;
        }

        for(long long c = 0; c < (1 << (2 * subsystem_size)); c++){
            int nonId = 0;
            for(int i = 0; i < subsystem_size; i++){
                nonId += ((c >> (2 * i)) & 3) != 0;
            }
            if(renyi_number_of_outcomes[c] >= 2)
                level_cnt[nonId] ++;
            level_ttl[nonId] ++;
        }

    Args:
        max_encoding (int): The maximum encoding value.
        subsystem_size (int): The size of the subsystem.
        all_number_of_outcomes (Sequence[int]): The number of outcomes for each encoding.

    Returns:
        tuple[list[int], list[int]]: The level count and total count for each level.
    """
    non_id_values = np.array([c_non_id(c, subsystem_size) for c in range(max_encoding)])
    outcomes_array = np.array(all_number_of_outcomes)

    level_ttl = np.bincount(non_id_values, minlength=subsystem_size + 1).tolist()

    valid_mask = outcomes_array >= 2
    level_cnt = np.bincount(
        non_id_values[valid_mask], minlength=subsystem_size + 1
    ).tolist()

    return level_cnt, level_ttl


def calculate_term_per_encoding(
    c: int,
    subsystem_size: int,
    sum_num_outcomes: int,
    binary_outcome: int,
    level_cnt: list[int],
    level_ttl: list[int],
) -> float:
    """Calculate the term for a given encoding in the entropy calculation.

    Original implementaion in Python by GitHub Copliot Claude Sonnet 4:

    .. code-block:: python

        if renyi_number_of_outcomes[c] <= 1:
            continue

        non_id = 0
        for i in range(subsystem_size):
            if ((c >> (2 * i)) & 3) != 0:
                non_id += 1

        if level_cnt[non_id] > 0:
            sum_squared = renyi_sum_of_binary_outcome[c] ** 2
            num_outcomes = renyi_number_of_outcomes[c]
            numerator = sum_squared - num_outcomes
            denominator = num_outcomes * (num_outcomes - 1)
            scale_factor = (
                level_ttl[non_id] / level_cnt[non_id] / (1 << subsystem_size)
            )
            term = numerator / denominator * scale_factor
            predicted_entropy += term

    Original implementaion in C++:

    .. code-block:: c++

        if(renyi_number_of_outcomes[c] <= 1) continue;

        int nonId = 0;
        for(int i = 0; i < subsystem_size; i++)
            nonId += ((c >> (2 * i)) & 3) != 0;

        predicted_entropy += ((double)1.0)
            / (renyi_number_of_outcomes[c] * (renyi_number_of_outcomes[c] - 1))
            * (
                renyi_sum_of_binary_outcome[c]
                * renyi_sum_of_binary_outcome[c]
                - renyi_number_of_outcomes[c]
            )
            / (1LL << subsystem_size)
            * level_ttl[nonId]
            / level_cnt[nonId];

    Args:
        c (int): The encoding to check.
        subsystem_size (int): The size of the subsystem.
        num_outcomes (int): The number of outcomes.
        sum_binary_outcome (int): The sum of binary outcome.
        level_cnt (list[int]): The count of levels.
        level_ttl (list[int]): The total time of levels.

    Returns:
        float: The calculated term for the given encoding.
    """
    if sum_num_outcomes <= 1:
        return 0.0

    non_id = c_non_id(c, subsystem_size)

    if level_cnt[non_id] <= 0:
        return 0.0

    return (
        (binary_outcome**2 - sum_num_outcomes)
        / (sum_num_outcomes * (sum_num_outcomes - 1))
        * (level_ttl[non_id] / level_cnt[non_id] / (1 << subsystem_size))
    )


def clamp_purity(purity: float, subsystem_size: int, resolution: float = 1e-9) -> float:
    """Clamp the purity value to be within the allowed range.

    Args:
        purity (float): The purity value to clamp.
        subsystem_size (int): The size of the subsystem.
        resolution (float): The resolution for clamping.

    Returns:
        float: The clamped purity value.
    """
    min_purity = 1.0 / (2.0**subsystem_size)
    max_purity = 1.0 - resolution
    return max(min(purity, max_purity), min_purity)


def predict_entropy(
    subsystem: list[int],
    pauli_basis: list[list[Literal[0, 1, 2]]],
    spin_outcome: list[list[Literal[1, -1]]],
) -> tuple[float, float, float]:
    """Predict the purity and the entropy of a quantum system.

    Args:
        subsystem (list[int]): The subsystems.
        pauli_basis (list[list[Literal[0, 1, 2]]]): The list of Pauli basis measurements.
        spin_outcome (list[list[Literal[1, -1]]]): The list of spin outcomes.

    Returns:
        tuple[float, float, float]:
            The predicted purity, clamped purity, and entropy of the quantum system.
    """
    subsystem_size = len(subsystem)
    max_encoding = 1 << (2 * subsystem_size)
    # Represent all possible comibinations of Pauli basis
    # Pauli: X: 1, Y: 2, Z: 3, I: 0 <-> X: 01, Y: 10, Z: 11, I: 00
    #
    # For example n = 8, 76543210, max_encoding = 2 ** (2 * subsystem_size)
    # For example c = 0b 0000 0001 1000 1100

    renyi_sum_of_binary_outcome = np.zeros(max_encoding, dtype=np.int64)
    # Record appearance times of all combinations of Pauli basis
    renyi_number_of_outcomes = np.zeros(max_encoding, dtype=np.int64)

    for single_pauli_base, single_spin_outcome in zip(pauli_basis, spin_outcome):
        encoding = 0  # encoding = 0b 0000 0000 0000 0000
        cumulative_outcome = 1

        renyi_sum_of_binary_outcome[0] += 1
        renyi_number_of_outcomes[0] += 1

        # Using gray code iteration over all 2^n possible outcomes
        # Gray Code: 0 -> 00, 1 -> 01, 2 -> 11, 3 -> 10
        # b in [1, 2^n-1]
        for b in range(1, 1 << subsystem_size):
            # For example n = 8, from 0b 0000 0001 to 0b 1111 1111
            # This is the bitstring from all possible measurement outcomes
            # ctz(01010101) -> 0, ctz(10000000) -> 7
            change_i = count_trailing_zeros(b)
            index_in_original_system = subsystem[change_i]
            # The clregs index

            cumulative_outcome *= single_spin_outcome[index_in_original_system]

            pauli_value = single_pauli_base[index_in_original_system]
            encoding ^= (pauli_value + 1) << (2 * change_i)
            # Gray Code: 0 -> 00, 1 -> 01, 2 -> 11, 3 -> 10
            # Pauli: I: 0, X: 1, Y: 2, Z: 3 <-> I: 00, X: 01, Y: 11, Z: 10
            # ----
            # For example ctz(01010101) -> 0 with X, ctz(10000000) -> 7 with Z
            #
            # ctz(01010101) -> 0 -> move 0  -> 0000 0000 0000 0001
            # ctz(10000000) -> 7 -> move 14 -> 1000 0000 0000 0000
            # ---
            # bitwise XOR, for example (0101) ^ (0011) = (0110)
            #
            # new_encoding_01: int = encoding ^ 0b 0000 0000 0000 0001
            # new_encoding_02: int = encoding ^ 0b 0000 0000 0000 0010

            renyi_sum_of_binary_outcome[encoding] += cumulative_outcome
            renyi_number_of_outcomes[encoding] += 1
            # binary_outcome[new_encoding_01] += (+1 or -1)
            # number_of_outcomes[new_encoding_01] += 1
            # binary_outcome[new_encoding_02] += (+1 or -1)
            # number_of_outcomes[new_encoding_02] += 1

    # Calculate level counts with optimized version
    level_cnt, level_ttl = calculate_level_count(
        max_encoding, subsystem_size, renyi_number_of_outcomes  # type: ignore
    )

    predicted_purity = sum(
        calculate_term_per_encoding(
            c,
            subsystem_size,
            renyi_number_of_outcomes[c],
            renyi_sum_of_binary_outcome[c],
            level_cnt,
            level_ttl,
        )
        for c in range(max_encoding)
    )

    clamped_purity = clamp_purity(predicted_purity, subsystem_size)
    entropy = -math.log2(clamped_purity)

    return predicted_purity, clamped_purity, entropy


class QuantumEntropyPredictor:
    """The executing instance."""

    def __init__(self):
        self.system_size = -1
        self.subsystems: list[list[int]] = []
        self.measurement_pauli_basis: list[list[Literal[0, 1, 2]]] = []
        self.measurement_spin_outcome: list[list[Literal[1, -1]]] = []

    def read_all_subsystems(self, subsystem_file_name: str):
        """
        讀取子系統文件並更新 subsystems 列表
        """
        self.subsystems, self.system_size = read_all_subsystems(subsystem_file_name)

    def read_all_measurements(self, measurement_file_name: str):
        """
        讀取測量文件並更新測量數據
        """
        (
            self.measurement_pauli_basis,  # type: ignore
            self.measurement_spin_outcome,  # type: ignore
            self.system_size,
        ) = read_all_measurements(measurement_file_name)

    def predict_entropy(self):
        """
        預測量子系統的熵
        """
        print(
            " ".join(
                map(
                    lambda x: f"{x}".rjust(16),
                    ["predicted_purity", "clamped_purity", "entropy"],
                )
            )
        )

        for s in self.subsystems:
            predicted_purity, clamped_purity, entropy = predict_entropy(
                s,
                self.measurement_pauli_basis,
                self.measurement_spin_outcome,
            )
            print(
                " ".join(
                    map(
                        lambda x: f"{x:f}".rjust(16),
                        [predicted_purity, clamped_purity, entropy],
                    )
                )
            )


def main():
    if len(sys.argv) != 4:
        print("Usage:", file=sys.stderr)
        usage_line = (
            "python prediction_shadow_entropy.py -e "
            "[measurement.txt] [subsystems.txt]"
        )
        print(usage_line, file=sys.stderr)
        print(
            "    This option predicts the entanglement entropy of subsystems.",
            file=sys.stderr,
        )
        sys.exit(-1)

    if sys.argv[1] != "-e":
        print("Error: Only -e option is supported in this version.", file=sys.stderr)
        sys.exit(-1)

    predictor = QuantumEntropyPredictor()
    predictor.read_all_measurements(sys.argv[2])
    predictor.read_all_subsystems(sys.argv[3])
    predictor.predict_entropy()


if __name__ == "__main__":
    main()
