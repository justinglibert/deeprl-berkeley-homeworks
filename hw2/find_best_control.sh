for b in 250 500 1000 2500 5000
    do
    for r in 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1
        do
        python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $b -lr $r -rtg --exp_name $(printf "hc_b%s_r%s" "$b" "$r")
    done
done 