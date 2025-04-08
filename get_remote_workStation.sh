sshpass -p "M@rt1n@CQT" scp -T martintrappe@172.18.120.183:~/Desktop/mpDPFT/mpDPFT.zip /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/WorkStation/
cd /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/WorkStation/
chmod +rwx *.*
unzip mpDPFT.zip
rm mpDPFT.zip
chmod +rwx *.*
FILE=/home/martintrappe/Desktop/PostDoc/Code/mpDPFT/WorkStation/mpDPFT_MovieData.tmp
if test -f "$FILE"; then
    mkdir Movie
    cp mpDPFT_MovieData.tmp mpDPFT_MovieData.dat
    cp mpDPFT_Movie.sh mpDPFT_MovieData.dat /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/WorkStation/Movie/
    cd /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/WorkStation/Movie/
    ./mpDPFT_Movie.sh
    cp mpDPFT_Movie.mp4 /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/WorkStation/
    cd /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/WorkStation/
    rm -R /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/WorkStation/Movie/
fi
FILE2=/home/martintrappe/Desktop/PostDoc/Code/mpDPFT/WorkStation/mpDPFT_OPLenergies.dat
if test -f "$FILE2"; then
    ./mpDPFT_OPLplots.sh
fi
./mpDPFT_Plots.sh
rm *.eps
chmod +rwx mpDPFT_CombinedPlots.tex
pdflatex mpDPFT_CombinedPlots.tex
rm mpDPFT_CombinedPlots.log
rm mpDPFT_CombinedPlots.aux
read -r VInterpolIdentifier < "mpDPFT_Aux.dat"
echo "VInterpolIdentifier=$VInterpolIdentifier" && rm mpDPFT_Aux.dat
TimeStamp="$(date +%Y%m%d_%H%M%S)" && echo "TimeStamp=$TimeStamp" && DirectoryName="mpDPFT_$TimeStamp-$VInterpolIdentifier"
mkdir /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/#DATA/#zips/$DirectoryName/
mv mpDPFT_V.dat mpDPFT_V_$TimeStamp.dat
rm mpDPFT_V_$VInterpolIdentifier.dat
cp mpDPFTmain.cpp mpDPFT.cpp mpDPFT.h MPDPFT_HEADER_*.h *.hpp mpDPFT.input Makefile Plugin*.* *.sh *.tex epslatex2epspdf *.dat *.pdf *.mp4 /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/#DATA/#zips/$DirectoryName/
zip "mpDPFT_SOURCE_$TimeStamp-$VInterpolIdentifier.zip" mpDPFTmain.cpp mpDPFT.cpp mpDPFT.h MPDPFT_HEADER_*.h *.hpp mpDPFT.input Makefile Plugin*.* *.sh *.tex epslatex2epspdf TabFunc_*.dat
cp mpDPFT_SOURCE_*.zip /home/martintrappe/Desktop/PostDoc/Code/mpDPFT/#Source_Backups
rm *.zip
if test -f "$FILE"; then
    rm mpDPFT_MovieData.tmp
    #okular mpDPFT_CombinedPlots.pdf & xdg-open mpDPFT_Movie.mp4
    okular mpDPFT_CombinedPlots.pdf & vlc mpDPFT_Movie.mp4
else okular mpDPFT_CombinedPlots.pdf
fi