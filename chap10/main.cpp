#include <llama.h>   // from the llama.cpp repository
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <resolv.h>
#include <sstream>
#include <string>
#include <vector>

#define BLUE "\033[1;34m"
#define GRAY "\033[1;30m"
#define RESET "\033[0m"

static const int g_ngpus         = 0;    // or some other number, like 99. for GPU usage
static const int g_ctxlen        = 2048; // adjust as needed
static const char *const g_mpath = "./models/qwen2.5-3b-instruct-q5_k_m.gguf";
static const float g_min_p       = 0.05f;
static const float g_temp        = 0.8f;

int main() {
    try {
        // log levels from llama.cpp, but only errors
        llama_log_set([](ggml_log_level level, const char * text, void * /* user_data */) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                fputs(text, stderr);
                fflush(stderr);
            }
        }, nullptr);

        // load dynamic backends
        ggml_backend_load_all();

        // load the model
        auto mp         = llama_model_default_params();
        mp.n_gpu_layers = g_ngpus;
        auto model      = llama_model_load_from_file(g_mpath, mp);
        if (!model) {
            throw std::runtime_error("Failed to load model from " + std::string{g_mpath});
        }
        const auto * vocab = llama_model_get_vocab(model);
        if (!vocab) {
            throw std::runtime_error("Failed to load vocab");
        }

        // context
        auto cp     = llama_context_default_params();
        cp.n_ctx    = g_ctxlen;
        cp.n_batch  = g_ctxlen;
        auto * ctx  = llama_init_from_model(model, cp);
        if (!ctx) {
            throw std::runtime_error("Failed to create context");
        }

        // sampler
        auto        rd   = std::random_device{};
        uint64_t    seed = (uint64_t(rd()) << 32) ^ rd();
        auto        sp   = llama_sampler_chain_default_params();
        auto *      smpl = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(g_min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(g_temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

        // generator loop
        auto generate = [&](const std::string & prompt) {
            std::stringstream response;

            auto memory         = llama_get_memory(ctx);
            const bool is_first = llama_memory_seq_pos_max(memory, 0) == -1;

            // tokenize the prompt
            // first, find out size of buffer
            const int n_tokens  = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, is_first, true);
            std::vector<llama_token> tokens(n_tokens);
            auto tok_res = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), is_first, true);
            if (tok_res < 0) {
                throw std::runtime_error("Failed to tokenize");
            }

            // batch
            auto batch = llama_batch_get_one(tokens.data(), tokens.size());
            llama_token token_id;

            printf(GRAY);
            while (true) {
                // enough space in context?
                auto n_ctx      = llama_n_ctx(ctx);
                memory          = llama_get_memory(ctx);
                auto n_ctx_used = llama_memory_seq_pos_max(memory, 0) + 1;
                if (n_ctx_used + batch.n_tokens > n_ctx) {
                    throw std::runtime_error("Exceeded conntex size");
                }

                // decode
                auto dec_res = llama_decode(ctx, batch);
                if (dec_res) {
                    throw std::runtime_error("Failed to decode");
                }

                // sample the next token
                token_id = llama_sampler_sample(smpl, ctx, - 1);
                if (llama_vocab_is_eog(vocab, token_id)) {
                    break;
                }

                // print and accumulate the token
                char buf[256];
                auto blen = llama_token_to_piece(vocab, token_id, buf, sizeof(buf)/sizeof(buf[0]), 0, true);
                if (blen < 0) {
                    throw std::runtime_error("Failed to detokenify");
                }
                printf("%.*s", blen, buf);
                fflush(stdout);
                response.write(buf, blen);

                // prepare the next batch with the sampled token
                batch = llama_batch_get_one(&token_id, 1);
            }
            printf(RESET"\n");

            return response.str();
        };

        std::vector<llama_chat_message> messages;
        std::vector<char> formatted(llama_n_ctx(ctx));
        int prev_len = 0;
        while (true) {
            std::string input;

            printf(BLUE "> ");
            std::getline(std::cin, input);
            printf(RESET);

            // leave if nothing is typed
            if (input.empty()) {
                break;
            }

            // obtain the template for model
            const auto * tmpl = llama_model_chat_template(model, /* name */ nullptr);

            // add the user input to the messages list
            messages.push_back({"user", strdup(input.c_str())});

            // apply the template
            auto new_len = llama_chat_apply_template(
                    tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
            if (new_len > (int)formatted.size()) {
                formatted.resize(new_len);
                new_len = llama_chat_apply_template(
                    tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
            }
            if (new_len < 0) {
                throw std::runtime_error("Failed to apply template");
            }

            // remove previous messages to obtain the prompt to generate the response
            std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);

            // generate a response
            auto response = generate(prompt);

            // add the response to the messages list
            messages.push_back({"assistant", strdup(response.c_str())});
            prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
            if (prev_len < 0) {
                throw std::runtime_error("Failed to apply template");
            }
        }

        // housekeeping
        for (auto & msg : messages) {
            free(const_cast<char *>(msg.content));
        }
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << "\n";
        return 1;
    }
}
