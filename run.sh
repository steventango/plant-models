#!/bin/bash

uv run python main.py --mode decoder --target_cols area --output_dir results/decoder_area --force --epochs 10
uv run python main.py --mode transition --decoder_checkpoint results/decoder_area/model --output_dir results/transition --force --epochs 10