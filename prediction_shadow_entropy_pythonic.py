#!/usr/bin/env python3
"""
Python version of prediction_shadow.cpp with more Pythonic code
and less mapping with the C++ implementation on variable names and structures.

Created by converting from Hsin-Yuan Huang's C++ implementation.
For more details, see the accompanying paper:
"Predicting Many Properties of a Quantum System from Very Few Measurements".

This version is converted by LLM agent,
Github Copilot with the model Claude Sonnet 4.

"""

import sys
import math
import numpy as np


def count_trailing_zeros(n: int) -> int:
    """
    計算二進制表示中尾隨零的數量 (等價於 C++ 的 __builtin_ctzll)
    """
    if n == 0:
        return 64  # 或其他適當的大數
    count = 0
    while (n & 1) == 0:
        n >>= 1
        count += 1
    return count


class QuantumEntropyPredictor:
    """The executing instance."""

    def __init__(self):
        self.system_size = -1
        self.subsystems = []
        self.measurement_pauli_basis = []
        self.measurement_binary_outcome = []

    def read_all_subsystems(self, subsystem_file_name: str):
        """
        讀取子系統文件並更新 subsystems 列表
        """
        try:
            with open(subsystem_file_name, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            error_msg = (
                f"\n====\nError: the input file "
                f'"{subsystem_file_name}" does not exist.\n====\n'
            )
            print(error_msg, file=sys.stderr)
            sys.exit(-1)

        # 讀取系統大小
        system_size_subsystem = int(lines[0].strip())
        if self.system_size == -1:
            self.system_size = system_size_subsystem

        # 逐行讀取局部觀測量
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            k_local = int(parts[0])

            ith_subsystem = []
            for k in range(k_local):
                position_of_the_qubit = int(parts[k + 1])
                ith_subsystem.append(position_of_the_qubit)

            self.subsystems.append(ith_subsystem)

    def read_all_measurements(self, measurement_file_name: str):
        """
        讀取測量文件並更新測量數據
        """
        try:
            with open(measurement_file_name, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            error_msg = (
                f"\n====\nError: the input file "
                f'"{measurement_file_name}" does not exist.\n====\n'
            )
            print(error_msg, file=sys.stderr)
            sys.exit(-1)

        # 讀取系統大小
        system_size_measurement = int(lines[0].strip())
        if self.system_size == -1:
            self.system_size = system_size_measurement
        if system_size_measurement != self.system_size:
            print(
                "\n====\nError: the system size do not match.\n====\n", file=sys.stderr
            )
            sys.exit(-1)

        # 逐行讀取測量數據
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            pauli_basis = []
            binary_outcome = []

            for i in range(self.system_size):
                pauli = parts[2 * i]
                outcome = int(parts[2 * i + 1])
                assert outcome in (1, -1)

                # 將 Pauli 字符轉換為數字 (X=0, Y=1, Z=2)
                pauli_basis.append(ord(pauli[0]) - ord("X"))
                binary_outcome.append(outcome)

            self.measurement_pauli_basis.append(pauli_basis)
            self.measurement_binary_outcome.append(binary_outcome)

    def predict_entropy(self):
        """
        預測量子系統的熵
        """
        print("purity   entropy")

        for s, _ in enumerate(self.subsystems):
            subsystem_size = len(self.subsystems[s])

            # 初始化 Renyi 數組
            max_encoding = 1 << (2 * subsystem_size)
            renyi_sum_of_binary_outcome = np.zeros(max_encoding, dtype=float)
            renyi_number_of_outcomes = np.zeros(max_encoding, dtype=float)

            # 對每個測量進行處理
            for t, _ in enumerate(self.measurement_pauli_basis):
                encoding = 0
                cumulative_outcome = 1

                renyi_sum_of_binary_outcome[0] += 1
                renyi_number_of_outcomes[0] += 1

                # 使用格雷碼迭代所有 2^n 種可能的結果
                for b in range(1, 1 << subsystem_size):
                    change_i = count_trailing_zeros(b)
                    index_in_original_system = self.subsystems[s][change_i]

                    cumulative_outcome *= self.measurement_binary_outcome[t][
                        index_in_original_system
                    ]
                    pauli_value = self.measurement_pauli_basis[t][
                        index_in_original_system
                    ]
                    encoding ^= (pauli_value + 1) << (2 * change_i)

                    renyi_sum_of_binary_outcome[encoding] += cumulative_outcome
                    renyi_number_of_outcomes[encoding] += 1

            # 計算層級計數
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

            # 計算預測熵
            predicted_entropy = 0.0
            for c in range(max_encoding):
                if renyi_number_of_outcomes[c] <= 1:
                    continue

                non_id = 0
                for i in range(subsystem_size):
                    if ((c >> (2 * i)) & 3) != 0:
                        non_id += 1

                if level_cnt[non_id] > 0:
                    # 計算熵項
                    num_outcomes = renyi_number_of_outcomes[c]
                    scale_factor = (
                        level_ttl[non_id] / level_cnt[non_id] / (1 << subsystem_size)
                    )
                    term = (
                        (renyi_sum_of_binary_outcome[c] ** 2 - num_outcomes)
                        / num_outcomes
                        * (num_outcomes - 1)
                        * scale_factor
                    )
                    predicted_entropy += term

            # 計算最終熵值
            min_purity = 1.0 / (2.0**subsystem_size)
            max_purity = 1.0 - 1e-9
            clamped_purity = max(min(predicted_entropy, max_purity), min_purity)
            entropy = -math.log2(clamped_purity)

            print(f"{predicted_entropy:.6f} {entropy:.6f}")


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
