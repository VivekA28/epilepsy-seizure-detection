#!/usr/bin/env bash
set -e
cd /home/vivek/Desktop/epilepsy-seizure-detection
exec systemd-inhibit --what=sleep:idle:handle-lid-switch --why="Multi-seed training" .venv/bin/python -u scripts/train_multi_seed_v2.py > train_multi_seed.log 2>&1
