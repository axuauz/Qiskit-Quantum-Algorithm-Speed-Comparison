# -*- coding: utf-8 -*-
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import os

# ==============================================================================
# SECTION 1: 양자 알고리즘 컴포넌트
# ==============================================================================

def create_braid_sub_circuit(num_qubits, braid_generator_index):
    """
    주어진 인덱스에 해당하는 꼬임군 생성자(Braid Group Generator) σ_i에 대한
    양자 회로(SWAP 게이트)를 생성합니다.
    """
    if not 0 <= braid_generator_index < num_qubits - 1:
        raise ValueError(f"Braid generator index {braid_generator_index} is out of bounds for {num_qubits} qubits.")
    
    braid_qc = QuantumCircuit(num_qubits, name=f'σ_{braid_generator_index}')
    braid_qc.swap(braid_generator_index, braid_generator_index + 1)
    return braid_qc

def create_dj_oracle(num_qubits, oracle_type='balanced'):
    """
    도이치-조사 알고리즘을 위한 오라클을 생성합니다.
    """
    oracle_qc = QuantumCircuit(num_qubits + 1, name='Oracle')
    
    if oracle_type == 'balanced':
        for i in range(num_qubits):
            oracle_qc.cx(i, num_qubits)
    elif oracle_type == 'constant':
        if np.random.randint(2) == 0:
            oracle_qc.x(num_qubits)
    else:
        raise ValueError("Oracle type must be 'constant' or 'balanced'.")
        
    return oracle_qc

def run_quantum_experiment(num_qubits, braid_sequence=None, oracle_type='balanced', shots=2048):
    """
    도이치-조사 알고리즘을 실행하고 시간, 결과, 회로 객체를 반환하는 양자 실험 함수입니다.
    """
    n = num_qubits
    qc = QuantumCircuit(n + 1, n)

    for i in range(n): qc.h(i)
    qc.x(n)
    qc.h(n)
    qc.barrier(label="Init")

    oracle = create_dj_oracle(n, oracle_type)
    qc.compose(oracle, inplace=True)
    qc.barrier(label="Oracle")

    if braid_sequence:
        qc.name = "DJ_with_Braid"
        for i in braid_sequence:
            braid_gate = create_braid_sub_circuit(n + 1, i)
            qc.compose(braid_gate, inplace=True)
        qc.barrier(label="Braid")
    else:
        qc.name = "Standard_DJ"

    for i in range(n): qc.h(i)
    qc.barrier(label="Final_H")
    qc.measure(range(n), range(n))

    simulator = AerSimulator()
    start_time = time.time()
    result = simulator.run(qc, shots=shots).result()
    end_time = time.time()
    
    execution_time = end_time - start_time
    counts = result.get_counts(qc)
    
    return execution_time, counts, qc

# ==============================================================================
# SECTION 2: 고전 알고리즘 컴포넌트
# ==============================================================================

def hanoi_solver_recursive(n, source, destination, auxiliary):
    """
    하노이의 탑 문제를 재귀적으로 해결하는 함수입니다. (I/O 시간 배제)
    """
    if n == 1:
        return 1
    count = 0
    count += hanoi_solver_recursive(n-1, source, auxiliary, destination)
    count += 1
    count += hanoi_solver_recursive(n-1, auxiliary, destination, source)
    return count

def run_classical_hanoi_experiment(num_disks):
    """
    하노이의 탑 알고리즘을 실행하고 시간을 측정하는 고전 실험 함수입니다.
    """
    print(f"하노이의 탑 문제 해결 시작 (원판 수: {num_disks})")
    print("연산 중...")

    start_time = time.time()
    total_moves = hanoi_solver_recursive(num_disks, 'A', 'C', 'B')
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    print(f"연산 완료. 총 이동 횟수: {total_moves} (이론값: {2**num_disks - 1})")
    return execution_time, total_moves

# ==============================================================================
# SECTION 3: 메인 실행 및 비교 분석
# ==============================================================================

if __name__ == '__main__':
    # --- 실험 설정 ---
    NUM_INPUT_QUBITS = 3
    ORACLE_TYPE = 'balanced'
    BRAID_OPERATIONS = [0, 1, 2] 
    NUM_DISKS_FOR_HANOI = 20

    # --- 실험 1: 표준 도이치-조사 알고리즘 ---
    print("--- 🔬 실험 1: 표준 도이치-조사 알고리즘 실행 ---")
    std_time, std_counts, std_qc = run_quantum_experiment(NUM_INPUT_QUBITS, braid_sequence=None, oracle_type=ORACLE_TYPE)
    print(f"[결과] 실행 시간: {std_time:.6f} 초")
    
    # [수정된 기능] 표준 회로도 텍스트 출력
    print("\n--- 표준 DJ 알고리즘 회로도 ---")
    print(std_qc)
    print("")

    # --- 실험 2: 꼬임군 적용 도이치-조사 알고리즘 ---
    print("--- 🔬 실험 2: 꼬임군 적용 도이치-조사 알고리즘 실행 ---")
    braid_time, braid_counts, braid_qc = run_quantum_experiment(NUM_INPUT_QUBITS, braid_sequence=BRAID_OPERATIONS, oracle_type=ORACLE_TYPE)
    print(f"[결과] 실행 시간: {braid_time:.6f} 초")

    # [수정된 기능] 꼬임군 적용 회로도 텍스트 출력
    print("\n--- 꼬임군 적용 DJ 알고리즘 회로도 ---")
    print(braid_qc)
    print("")

    # --- 실험 3: 고전 하노이의 탑 알고리즘 ---
    print("--- 🔬 실험 3: 고전 하노이의 탑 알고리즘 실행 ---")
    hanoi_time, hanoi_moves = run_classical_hanoi_experiment(NUM_DISKS_FOR_HANOI)
    print(f"[결과] 실행 시간: {hanoi_time:.6f} 초\n")

    # --- 최종 종합 비교 분석 ---
    print("\n\n" + "="*60)
    print("          📊 최종 실험 결과 종합 비교 분석 📊")
    print("="*60)
    print("이 분석은 각기 다른 패러다임의 알고리즘 실행 시간을 비교하여\n컴퓨팅 자원 소모의 스케일 차이를 확인하는 것을 목표로 합니다.\n")
    
    print(f" [양자 실험 조건]")
    print(f"  - 입력 큐비트: {NUM_INPUT_QUBITS}개")
    print(f"  - 꼬임 연산 수: {len(BRAID_OPERATIONS) if BRAID_OPERATIONS else 0}개")
    print(f"\n [고전 실험 조건]")
    print(f"  - 하노이 원판 수: {NUM_DISKS_FOR_HANOI}개\n")

    print("-" * 60)
    print(" [실행 시간 비교]")
    print(f"  1. 표준 양자 알고리즘 (Qiskit):        {std_time:.6f} 초")
    print(f"  2. 꼬임군 적용 양자 알고리즘 (Qiskit): {braid_time:.6f} 초")
    print(f"  3. 고전 재귀 알고리즘 (Python):        {hanoi_time:.6f} 초")
    print("-" * 60)
    
    time_diff = braid_time - std_time
    performance_change = (time_diff / std_time) * 100 if std_time > 0 else 0

    print("\n [양자 알고리즘 내 성능 분석]")
    if performance_change >= 0:
        print(f"  - 꼬임 연산으로 인해 실행 시간이 {performance_change:.2f}% 증가했습니다.")
    else:
        print(f"  - 꼬임 연산으로 인해 실행 시간이 {-performance_change:.2f}% 감소 (최적화).")

    # --- 결과 히스토그램 시각화 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_histogram(std_counts, ax=axes[0], title='Standard DJ Algorithm Results')
    plot_histogram(braid_counts, ax=axes[1], title=f'Braid-Applied DJ Results ({len(BRAID_OPERATIONS)} ops)')
    plt.suptitle("Quantum Simulation Measurement Results", fontsize=16)
    
    hist_filename = 'dj_braid_histogram_comparison.png'
    plt.savefig(hist_filename)
    print(f"\n✅ 양자 실험 비교 히스토그램이 '{hist_filename}' 파일로 저장되었습니다.")
