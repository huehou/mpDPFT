
#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-CutDen0/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,6.875
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set title 'Den0 along (-6,0)\$\to\$(6,0)'
set mxtics 5
set mytics 5
set grid xtics ytics mxtics mytics
set yrange [0.000000:2.32723]
mu(x)=1.000000
plot 'mpDPFT_CutData.dat' using 1:2 with lines ls 1 title 'Den0'
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-CutEnv+V0/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,6.875
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set title 'Env+V0 along (-6,0)\$\to\$(6,0)'
set mxtics 5
set mytics 5
set grid xtics ytics mxtics mytics
set yrange [0.000000:1.1]
mu(x)=1.000000
plot 'mpDPFT_CutData.dat' using 1:3 with lines ls 3 title 'Env0', mu(x) ls 2 title '\$\mu\$', 'mpDPFT_CutData.dat' using 1:4 with lines ls 7 title 'V0' 
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo

#!/bin/bash
GNUPLOT=gnuplot
OUTPUT=`echo $0 | sed 's/\.sh/-LogPlot-CutDen0/'`
$GNUPLOT << EOF
set terminal epslatex dashed size 10,6.875
set output "tmp_split.tex"
xunitsize=0.30
yunitsize=0.30
set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0
set size xunitsize,yunitsize
set origin 0,0
set style line 1 dt 1 lw 6 lc rgb "#0000ff"
set style line 2 dt (5,2) lw 6 lc rgb "#ffa500"
set style line 3 dt (1,1) lw 6 lc rgb "#00ff00"
set style line 4 dt (5,2) lw 6 lc rgb "#a9a9a9"
set style line 5 dt 1 lw 6 lc rgb "#000000"
set style line 6 dt 1 lw 6 lc rgb "#00ff00"
set style line 7 dt 1 lw 2 lc rgb "#ff0000"
set style line 11 dt 1 lw 2 lc rgb "#ff0000"
set style line 12 dt (1,2) lw 2 lc rgb "#0000ff"
set style line 13 dt (4,1) lw 6 lc rgb "#0000ff"
set style line 14 dt (1,2) lw 2 lc rgb "#00ff00"
set style line 15 dt (4,1) lw 6 lc rgb "#00ff00"
set title 'LogPlotDen0 along (-6,0)\$\to\$(6,0)'
set logscale y
set format y "%e"
set samples 1000
set yrange [2.35958e-10:2.32723]
mu(x)=1.000000
plot 'mpDPFT_CutData.dat' using 1:2 with lines ls 1 title 'Den0'
EOF
cat tmp_split.eps \
  | sed 's/\(\/LT2.\+\[\).\+\(\] LC2.\+def\)/\1 2 dl1 3 dl2 \2/' \
  | sed 's/\(\/LT1.\+\[\).\+\(\] LC1.\+def\)/\1 6 dl1 6 dl2 \2/' \
  > tmp.eps && mv tmp.eps tmp_split.eps
cat tmp_split.tex \
  | sed 's/\$\(1\)e+00\$/\$1\$/g' \
  | sed 's/\$\(.\+\)e+\(.\+\)\$/\$\1 \\cdot 10\^{\2}\$/g' \
  | sed 's/\$\(1\)e-0*\(.\+\)\$/\$10\^{-\2}\$/g' \
  | sed 's/\$\(.\+\)e-0*\(.\+\)\$/\$\1 \\cdot 10\^{-\2}\$/g' \
  | sed 's/10\^{0/10\^{/g' \
  > tmp.tex && mv tmp.tex tmp_split.tex &&\
epslatex2epspdf tmp_split $OUTPUT &&\
rm -f tmp_split.tex tmp_split.eps &&\
echo "Generated $OUTPUT.eps and $OUTPUT.pdf" && echo
