// frame_walker.cpp

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

static void emit_dimensions(int width, int height) {
    // Send stream header: width (uint32_t), height (uint32_t)
    uint32_t W = static_cast<uint32_t>(width);
    uint32_t H = static_cast<uint32_t>(height);

    std::cout.write(reinterpret_cast<char*>(&W), sizeof(W));
    std::cout.write(reinterpret_cast<char*>(&H), sizeof(H));
    std::cout.flush();
}

static void serialize(std::vector<float> &tensor) {
    uint64_t tensor_bytes = tensor.size() * sizeof(float);

    // header: number of bytes
    std::cout.write(reinterpret_cast<char*>(&tensor_bytes), sizeof(tensor_bytes));

    // body: raw float tensor
    std::cout.write(reinterpret_cast<char*>(tensor.data()), tensor_bytes);
    std::cout.flush(); // ensure Python receives the frame
}

static void emit_tensor(uint8_t *data, int w, int h) {
    // Allocate tensor: 3 channels of H x W floats.
    std::vector<float> tensor(3 * h * w);

    uint8_t* rgb = data;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int rgb_index = 3 * (y * w + x);

            float R = rgb[rgb_index + 0] / 255.0f;
            float G = rgb[rgb_index + 1] / 255.0f;
            float B = rgb[rgb_index + 2] / 255.0f;

            tensor[0 * h * w + y * w + x] = R;
            tensor[1 * h * w + y * w + x] = G;
            tensor[2 * h * w + y * w + x] = B;
        }
    }

    serialize(tensor);
}

static void fail(const std::string& msg, int err) {
    char buf[256];
    av_strerror(err, buf, sizeof(buf));
    throw std::runtime_error(msg + ": " + buf);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " input.mp4\n";
        return 1;
    }

    const char* filename = argv[1];

    // 1. Open container and read stream info.
    AVFormatContext* fmt_ctx = nullptr;
    int ret = avformat_open_input(&fmt_ctx, filename, nullptr, nullptr);
    if (ret < 0) fail("avformat_open_input", ret);

    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) fail("avformat_find_stream_info", ret);

    // 2. Find the best video stream.
    int video_stream_index = av_find_best_stream(
        fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_index < 0)
        fail("Could not find video stream", video_stream_index);

    AVStream*      video_stream = fmt_ctx->streams[video_stream_index];
    AVCodecParameters* codecpar = video_stream->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec)
        throw std::runtime_error("Unsupported codec");

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx)
        throw std::runtime_error("avcodec_alloc_context3 failed");

    ret = avcodec_parameters_to_context(codec_ctx, codecpar);
    if (ret < 0) fail("avcodec_parameters_to_context", ret);

    ret = avcodec_open2(codec_ctx, codec, nullptr);
    if (ret < 0) fail("avcodec_open2", ret);

    // 3. Prepare scaling context for RGB24.
    SwsContext* sws_ctx = sws_getContext(
        codec_ctx->width,
        codec_ctx->height,
        codec_ctx->pix_fmt,
        codec_ctx->width,
        codec_ctx->height,
        AV_PIX_FMT_RGB24,
        SWS_BILINEAR,
        nullptr, nullptr, nullptr);

    if (!sws_ctx)
        throw std::runtime_error("sws_getContext failed");

    AVFrame* frame      = av_frame_alloc();   // decoded frame (native fmt)
    AVFrame* rgb_frame  = av_frame_alloc();   // converted frame (RGB24)
    AVPacket* pkt       = av_packet_alloc();

    if (!frame || !rgb_frame || !pkt)
        throw std::runtime_error("Could not allocate frame/packet");

    // Allocate buffer for RGB frame.
    int rgb_buf_size = av_image_get_buffer_size(
        AV_PIX_FMT_RGB24, codec_ctx->width, codec_ctx->height, 1);

    uint8_t* rgb_buffer = (uint8_t*)av_malloc(rgb_buf_size);
    if (!rgb_buffer)
        throw std::runtime_error("av_malloc failed");

    av_image_fill_arrays(
        rgb_frame->data,
        rgb_frame->linesize,
        rgb_buffer,
        AV_PIX_FMT_RGB24,
        codec_ctx->width,
        codec_ctx->height,
        1);

    int64_t frame_index = 0;
    AVRational time_base = video_stream->time_base;

    emit_dimensions(
        codec_ctx->width,
        codec_ctx->height);

    std::cerr << "Walking frames in " << filename << "...\n";

    // 4. Read packets, send to decoder, receive frames.
    while ((ret = av_read_frame(fmt_ctx, pkt)) >= 0) {
        if (pkt->stream_index == video_stream_index) {
            ret = avcodec_send_packet(codec_ctx, pkt);
            if (ret < 0) fail("avcodec_send_packet", ret);

            while ((ret = avcodec_receive_frame(codec_ctx, frame)) >= 0) {
                // Convert to RGB24.
                sws_scale(
                    sws_ctx,
                    frame->data,
                    frame->linesize,
                    0,
                    codec_ctx->height,
                    rgb_frame->data,
                    rgb_frame->linesize);

                // Compute timestamp in seconds (if available).
                double ts_sec = 0.0;
                if (frame->best_effort_timestamp != AV_NOPTS_VALUE) {
                    ts_sec = frame->best_effort_timestamp *
                             av_q2d(time_base);
                }

                std::cerr << "Frame " << frame_index++
                          << " @ " << ts_sec << " s, "
                          << codec_ctx->width << "x"
                          << codec_ctx->height << "\n";

                // At this point, rgb_frame->data[0] is a contiguous
                // RGB24 buffer of size rgb_buf_size bytes.
                // We will feed this into the model in the next section.

                // Convert to tensor
                emit_tensor(
                    rgb_frame->data[0],
                    codec_ctx->width,
                    codec_ctx->height);
            }

            if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
                fail("avcodec_receive_frame", ret);
            }
        }

        av_packet_unref(pkt);
    }

    // Flush decoder.
    avcodec_send_packet(codec_ctx, nullptr);
    while ((ret = avcodec_receive_frame(codec_ctx, frame)) >= 0) {
        // (Optional) handle remaining frames.
    }

    std::cerr << "Done. Processed " << frame_index << " frames.\n";

    // 5. Cleanup.
    av_free(rgb_buffer);
    av_frame_free(&rgb_frame);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    sws_freeContext(sws_ctx);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);

    return 0;
}
