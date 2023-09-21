for f in test_results/DeepLens/*/pred_*.mp4; do
    mkdir frames
    ffmpeg -i "$f" frames/out-%03d.jpg
    convert -delay 20 frames/*.jpg "${f%.*}.gif"
    rm -r frames
done