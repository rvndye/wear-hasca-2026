#!/bin/zsh
# Pull the public Kaggle notebooks this project builds on into endgame/ref_public/ (not tracked in git;
# each notebook keeps its author's license - Kaggle public notebooks are Apache-2.0 unless stated otherwise).
# Needs the Kaggle CLI and a Kaggle API token in ~/.kaggle/access_token (see README).
export KAGGLE_USERNAME=x KAGGLE_KEY=x KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
cd "${0:A:h}" && mkdir -p ref_public
for k in goodpjw2008/wear-hasca-part-2-ranked-links-lb-0-92990 goodpjw2008/wear-hasca-learned-links-counts-lb-0-92942 \
         woominyo/lb-0-890-wear-hasca-2026-timeline-graph honghanhhh/wear-hasca-lb-0-9 \
         yeashusemwal/ranked-timelines-boundary-refiner-lb-0-928 sujanmajhisuzan/wear-hasca-sota-bagged-boundary-refiner \
         jiweiliu/public-fast-gpu-inference jiweiliu/wear-soft-context-private-inference \
         kansukehabano/what-egocentric-video-embeddings-encode avikdas567/multimodal-videomae-inertial-cross-attention-har; do
  d=ref_public/${k#*/}; mkdir -p $d; kaggle kernels pull $k -p $d -m 2>&1 | grep -v -i "warn\|urllib" | tail -1
done
