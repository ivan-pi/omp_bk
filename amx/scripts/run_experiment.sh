{ echo "p E GDoF/s GB/s"
  for p in 1 2 3 4 5 6 7 8; do
    echo "$p scalar $(./bk1 $p 100000 3 | sed -n 's/.*GDoF\/s = \([0-9.e-]*\) GB\/s = \([0-9.e-]*\).*/\1 \2/p')"
    for E in 1 2 4 8 16 32; do
      echo "$p $E $(BK_BATCH=$E ./bk1_amx $p 100000 3 | sed -n 's/.*GDoF\/s = \([0-9.e-]*\) GB\/s = \([0-9.e-]*\).*/\1 \2/p')"
    done
  done
} | column -t