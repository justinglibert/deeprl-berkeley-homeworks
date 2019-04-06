configs=();
folders=()
for b in 250 500 1000 2500
    do
    for r in 1e-3 5e-3 1e-2
        do
        folders=("${folders[@]}" $(printf "data/hc_b%s_r%s_InvertedPendulum-v2" "$b" "$r")) 
        configs=("${configs[@]}" $(printf "hc_b%s_r%s" "$b" "$r")) 
    done
done
python.app plot.py ${folders[@]} --value AverageReturn --legend ${configs[@]} --save_name p5