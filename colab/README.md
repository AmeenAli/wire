# WIRE in a minute (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmeenAli/wire/blob/main/colab/wire_tour.ipynb)

`wire_tour.ipynb` reads a short clip, or your own video, the way WIRE does. It shows the decoder input of the frozen backbone and the refined tokens that replace each frame pair. It then shows how much smaller the input and the cache become. On a GPU with at least 24 GB, it also asks the uncompressed backbone, WIRE and WIRE without its update the same question.

On a free CPU runtime the tour takes about half a minute. It downloads the Qwen3-VL video processor (about 15 MB, no model weights) and the sample clip (7 MB). It uses the repository's own input preparation (`wire/inference.py`) and contraction rule (`wire/compressor.py`). `wire_tour.py` draws the figures, the animation and the comparison table, and `paper_results.json` holds the paper's measurements that the tour quotes.

The setup cell takes the path or URL of your own video, a question, and the number of frames to sample.

The sample clip is "Peacock walking and eating" by Mx. Granger on Wikimedia Commons, released under CC0 1.0 (https://commons.wikimedia.org/wiki/File:Peacock_walking_and_eating.webm).

Project page: https://ameenali.github.io/wire/
