#!/usr/bin/env bash

./llama.cpp/build/bin/llama-cli \
    -m models/qwen2.5-3b-instruct-q5_k_m.gguf \
    -c 8192 \
    -ngl 0 \
    -dev Vulkan0

