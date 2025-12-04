c++ \
  --std c++20 \
  -I./llama.cpp/include \
  -I./llama.cpp/ggml/include \
  -L./llama.cpp/build/bin \
  -lllama \
  -lggml \
  -Wl,-rpath,'$ORIGIN/llama.cpp/build/bin' \
  main.cpp \
  -o main
