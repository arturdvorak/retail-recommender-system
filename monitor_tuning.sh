#!/bin/bash
# Real-time progress monitor for SVD hyperparameter tuning

clear
echo "============================================================"
echo "SVD Hyperparameter Tuning - Real-time Progress Monitor"
echo "============================================================"
echo "Configuration: 15 trials, 3,000 users per trial"
echo "Press Ctrl+C to stop monitoring"
echo "============================================================"
echo ""

while true; do
    clear
    echo "============================================================"
    echo "SVD Hyperparameter Tuning - Real-time Progress"
    echo "============================================================"
    date
    echo ""
    
    # Check process
    PROC=$(ps aux | grep "hyperparameter_tuning.py.*svd" | grep -v grep | head -1)
    if [ -z "$PROC" ]; then
        echo "✗ Process NOT running (may have completed)"
    else
        PID=$(echo $PROC | awk '{print $2}')
        CPU=$(echo $PROC | awk '{print $3}')
        MEM=$(echo $PROC | awk '{print $4}')
        RUNTIME=$(echo $PROC | awk '{print $10}')
        echo "✓ Process Running (PID: $PID)"
        echo "  CPU: ${CPU}% | Memory: ${MEM}% | Runtime: $RUNTIME"
    fi
    
    echo ""
    echo "--- Trial Progress ---"
    
    # Count completed trials from log
    TRIALS=$(grep -c "Trial [0-9]\+/15:" svd_tuning.log 2>/dev/null || echo "0")
    echo "Completed trials: $TRIALS / 15"
    
    # Show latest trial results
    if [ -f svd_tuning.log ]; then
        echo ""
        echo "Latest trial results:"
        grep "Trial [0-9]\+/15:" svd_tuning.log | tail -3 | sed 's/^/  /'
        
        # Show best so far
        BEST=$(grep "Best trial:" svd_tuning.log | tail -1)
        if [ ! -z "$BEST" ]; then
            echo ""
            echo "Best trial so far:"
            echo "$BEST" | sed 's/^/  /'
        fi
    fi
    
    # Check MLflow runs
    MLFLOW_COUNT=$(find mlruns -type d -name "trial_*" -path "*/SVD_Recommendation_System/*" 2>/dev/null | wc -l | tr -d ' ')
    echo ""
    echo "MLflow runs: $MLFLOW_COUNT"
    
    echo ""
    echo "============================================================"
    echo "Refreshing in 5 seconds... (Ctrl+C to stop)"
    sleep 5
done
