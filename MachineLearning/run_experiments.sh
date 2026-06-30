#!/bin/bash
set -e

EXE="./main.exe"
RESULTS="results.txt"
LOG_DIR="experiment_logs"

mkdir -p "$LOG_DIR"

# clear previous results
echo "=== D2L 4.2.7 + 4.3.3 实验开始 ===" > "$RESULTS"
echo "时间: $(date)" >> "$RESULTS"
echo "" >> "$RESULTS"

run_exp() {
    local name="$1"; shift
    echo "==============================================" | tee -a "$RESULTS"
    echo ">>> $name" | tee -a "$RESULTS"
    echo "    命令: $EXE $*" | tee -a "$RESULTS"
    echo "    开始: $(date +%H:%M:%S)" | tee -a "$RESULTS"
    "$EXE" "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    local acc
    acc=$(grep "Test accuracy:" "$LOG_DIR/${name}.log" | tail -1)
    echo "    $acc" | tee -a "$RESULTS"
    echo "" >> "$RESULTS"
}

START_TIME=$(date +%s)

echo ""
echo "============================"
echo "  习题 1: 变化 num_hiddens"
echo "============================"

run_exp "ex1_h64"   --hiddens 64   --epochs 10 --prefix ex1_h64
run_exp "ex1_h128"  --hiddens 128  --epochs 10 --prefix ex1_h128
run_exp "ex1_h256"  --hiddens 256  --epochs 10 --prefix ex1_h256
run_exp "ex1_h512"  --hiddens 512  --epochs 10 --prefix ex1_h512
run_exp "ex1_h1024" --hiddens 1024 --epochs 10 --prefix ex1_h1024

echo ""
echo "============================"
echo "  习题 2: 增加隐藏层"
echo "============================"

run_exp "ex2_l1" --hiddens 256 --layers 1 --epochs 10 --prefix ex2_l1
run_exp "ex2_l2" --hiddens 256 --layers 2 --epochs 10 --prefix ex2_l2
run_exp "ex2_l3" --hiddens 256 --layers 3 --epochs 10 --prefix ex2_l3

echo ""
echo "============================"
echo "  习题 3: 变化学习率"
echo "============================"

run_exp "ex3_lr001" --lr 0.001 --epochs 10 --prefix ex3_lr001
run_exp "ex3_lr01"  --lr 0.01  --epochs 10 --prefix ex3_lr01
run_exp "ex3_lr05"  --lr 0.05  --epochs 10 --prefix ex3_lr05
run_exp "ex3_lr1"   --lr 0.1   --epochs 10 --prefix ex3_lr1
run_exp "ex3_lr5"   --lr 0.5   --epochs 10 --prefix ex3_lr5
run_exp "ex3_lr10"  --lr 1.0   --epochs 10 --prefix ex3_lr10

echo ""
echo "============================"
echo "  习题 4: 联合优化"
echo "============================"

run_exp "ex4_combo1"  --lr 0.05 --hiddens 256 --layers 1 --epochs 15 --prefix ex4_combo1
run_exp "ex4_combo2"  --lr 0.05 --hiddens 256 --layers 2 --epochs 15 --prefix ex4_combo2
run_exp "ex4_combo3"  --lr 0.05 --hiddens 512 --layers 1 --epochs 15 --prefix ex4_combo3
run_exp "ex4_combo4"  --lr 0.05 --hiddens 512 --layers 2 --epochs 15 --prefix ex4_combo4
run_exp "ex4_combo5"  --lr 0.1  --hiddens 256 --layers 1 --epochs 20 --prefix ex4_combo5
run_exp "ex4_combo6"  --lr 0.1  --hiddens 256 --layers 2 --epochs 20 --prefix ex4_combo6
run_exp "ex4_combo7"  --lr 0.1  --hiddens 512 --layers 1 --epochs 20 --prefix ex4_combo7
run_exp "ex4_combo8"  --lr 0.1  --hiddens 512 --layers 2 --epochs 20 --prefix ex4_combo8
run_exp "ex4_combo9"  --lr 0.01 --hiddens 128 --layers 1 --epochs 30 --prefix ex4_combo9
run_exp "ex4_combo10" --lr 0.01 --hiddens 256 --layers 2 --epochs 30 --prefix ex4_combo10

echo ""
echo "============================"
echo "  习题 4.3.3-2: 不同激活函数"
echo "============================"

run_exp "ex5_relu"    --activation relu    --epochs 10 --prefix ex5_relu
run_exp "ex5_sigmoid" --activation sigmoid --epochs 10 --prefix ex5_sigmoid

echo ""
echo "============================"
echo "  习题 4.3.3-3: 不同权重初始化"
echo "============================"

run_exp "ex6_custom" --init custom --epochs 10 --prefix ex6_custom
run_exp "ex6_normal" --init normal --epochs 10 --prefix ex6_normal
run_exp "ex6_xavier" --init xavier --epochs 10 --prefix ex6_xavier
run_exp "ex6_he"     --init he     --epochs 10 --prefix ex6_he

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo "" | tee -a "$RESULTS"
echo "==============================================" | tee -a "$RESULTS"
echo "  全部实验完成!" | tee -a "$RESULTS"
echo "  总耗时: ${ELAPSED_MIN}分${ELAPSED_SEC}秒" | tee -a "$RESULTS"
echo "  结束时间: $(date)" | tee -a "$RESULTS"
echo "==============================================" | tee -a "$RESULTS"
