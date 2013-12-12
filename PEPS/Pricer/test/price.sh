
PRICER="/Users/jl/live/research/src/mc_pricer/src/MC_RM"
DATA=$(ls *.dat)

for f in $DATA; do
    echo "model type bs" > "$f-tmp"
    cat "$f" >> "$f-tmp"
    echo "dividend rate 0." >> "$f-tmp"
    echo "FD step 0.1" >> "$f-tmp"
    echo "step size 1" >> "$f-tmp"
    echo "------------------------------------------"
    echo "$f"
    "$PRICER" --reduced --sample-average "$f-tmp"
    rm -f "$f-tmp"
done

