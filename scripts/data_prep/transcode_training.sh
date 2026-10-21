#!/usr/bin/env bash
# Transcode the 4K HDR (HEVC 10-bit, HLG/BT.2020) training videos to what the
# challenge videos look like: 1280x720 SDR BT.709 H.264 at ~3 Mbps.
# Hable tone-mapping matched the challenge colour look best of the variants tried
# (naive 8-bit decode is desaturated, direct HLG->709 clips). Frame count is kept
# exactly (fps_mode passthrough) so GT frame ids stay valid.
set -euo pipefail
FF=${FFMPEG:-$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")}
SRC=${1:-/data/trackid3x3/training/videos}
DST=${2:-/data/trackid3x3/training/videos_720p}
JOBS=${JOBS:-2}
mkdir -p "$DST"
VF="zscale=w=1280:h=720:filter=lanczos,zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
ls "$SRC"/*.MOV | xargs -P "$JOBS" -I{} bash -c '
  name=$(basename {} .MOV)
  [ -f "'"$DST"'/$name.mp4" ] && exit 0
  "'"$FF"'" -hide_banner -loglevel error -y -i {} -an -vf "'"$VF"'" -c:v libx264 -preset medium \
    -b:v 3000k -maxrate 4000k -bufsize 6000k -fps_mode passthrough "'"$DST"'/$name.tmp.mp4" \
  && mv "'"$DST"'/$name.tmp.mp4" "'"$DST"'/$name.mp4" && echo "done $name"
'
