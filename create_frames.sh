cd ./sample_videos
for f in *.mkv;do ffmpeg -i "$f" "${f%.mkv}"_%d.png;done