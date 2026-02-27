#include <sesg/stream_rtsp.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
#include <rtsp.h>

#include <app_ipcam_comm.h>
#include <app_ipcam_ll.h>
#include <app_ipcam_venc.h>
}

// Optional SG2002/CV18xx RGN/OSD support.
// This repo doesn't vendor SDK headers, so we use __has_include and compile-time guards.
#if defined(SESG_STREAM_RTSP_ENABLE_RGN)
#if __has_include(<cvi_region.h>)
#include <cvi_region.h>
#define SESG_STREAM_RTSP_HAS_RGN 1
#elif __has_include(<cvi_rgn.h>)
#include <cvi_rgn.h>
#define SESG_STREAM_RTSP_HAS_RGN 1
#else
#define SESG_STREAM_RTSP_HAS_RGN 0
#endif
#else
#define SESG_STREAM_RTSP_HAS_RGN 0
#endif

namespace sesg::stream_rtsp {

namespace {

static inline float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static inline uint16_t argb8888_to_argb1555(uint32_t argb) {
    const uint8_t a8 = (argb >> 24) & 0xFF;
    const uint8_t r8 = (argb >> 16) & 0xFF;
    const uint8_t g8 = (argb >> 8) & 0xFF;
    const uint8_t b8 = (argb >> 0) & 0xFF;

    const uint16_t a1 = (a8 >= 128) ? 0x8000u : 0x0000u;
    const uint16_t r5 = static_cast<uint16_t>((r8 >> 3) & 0x1Fu);
    const uint16_t g5 = static_cast<uint16_t>((g8 >> 3) & 0x1Fu);
    const uint16_t b5 = static_cast<uint16_t>((b8 >> 3) & 0x1Fu);
    return static_cast<uint16_t>(a1 | (r5 << 10) | (g5 << 5) | (b5));
}

static void draw_rect_outline_argb1555(std::vector<uint16_t>& buf, uint32_t w, uint32_t h, int x1, int y1, int x2,
                                       int y2, uint16_t color, uint16_t thickness) {
    if (w == 0 || h == 0) return;
    if (x2 <= x1 || y2 <= y1) return;

    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 > static_cast<int>(w)) x2 = static_cast<int>(w);
    if (y2 > static_cast<int>(h)) y2 = static_cast<int>(h);
    if (x2 <= x1 || y2 <= y1) return;

    const int t = std::max<int>(1, thickness);

    auto set_px = [&](int x, int y) {
        if (static_cast<uint32_t>(x) >= w || static_cast<uint32_t>(y) >= h) return;
        buf[static_cast<size_t>(y) * w + static_cast<size_t>(x)] = color;
    };

    // top/bottom
    for (int dy = 0; dy < t; ++dy) {
        int yt = y1 + dy;
        int yb = y2 - 1 - dy;
        for (int x = x1; x < x2; ++x) {
            set_px(x, yt);
            set_px(x, yb);
        }
    }

    // left/right
    for (int dx = 0; dx < t; ++dx) {
        int xl = x1 + dx;
        int xr = x2 - 1 - dx;
        for (int y = y1; y < y2; ++y) {
            set_px(xl, y);
            set_px(xr, y);
        }
    }
}

} // namespace

static inline uint64_t now_ms_steady() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static std::string makeSessionPath(const ServerConfig& server_cfg, const ChannelConfig& ch_cfg) {
    std::string p = ch_cfg.path;
    while (!p.empty() && p.front() == '/') {
        p.erase(p.begin());
    }
    if (!p.empty()) {
        return p;
    }
    if (!server_cfg.default_path_prefix.empty()) {
        return server_cfg.default_path_prefix + std::to_string(static_cast<int>(ch_cfg.ch));
    }
    return std::string("ch") + std::to_string(static_cast<int>(ch_cfg.ch));
}

static CVI_RTSP_VIDEO_CODEC toRtspVideoCodec(video_format_t format) {
    switch (format) {
        case VIDEO_FORMAT_H264:
            return RTSP_VIDEO_H264;
        case VIDEO_FORMAT_H265:
            return RTSP_VIDEO_H265;
        case VIDEO_FORMAT_JPEG:
            return RTSP_VIDEO_JPEG;
        default:
            return RTSP_VIDEO_NONE;
    }
}

struct SesgRtspVpssStreamer::Impl {
    std::atomic<bool> running{false};

    bool attached_mode{false};
    int consumer_index{0};
    std::vector<ChannelConfig> channels;

    std::mutex mu;
    CVI_RTSP_CTX* rtsp_ctx{nullptr};
    std::unordered_map<int, CVI_RTSP_SESSION*> sessions_by_venc_chn;

    struct OverlayState {
        bool enabled{false};
        uint32_t w{0};
        uint32_t h{0};
        std::vector<OverlayRect> rects;
        bool dirty{false};

        // RGN resources are created lazily in attached mode.
        bool rgn_ready{false};
        int rgn_handle{-1};
        std::vector<uint16_t> canvas_argb1555;

        // Avoid retry storm when RGN driver/resources are unavailable.
        uint32_t rgn_fail_count{0};
        uint64_t rgn_next_retry_ms{0};
    };
    std::unordered_map<int, OverlayState> overlay_by_venc_chn;

    bool vpss_inited{false};

    static void onConnect(const char* ip, void* /*arg*/) {
        APP_PROF_LOG_PRINT(LEVEL_INFO, "rtsp client connected: %s\n", ip);
    }

    static void onDisconnect(const char* ip, void* /*arg*/) {
        APP_PROF_LOG_PRINT(LEVEL_INFO, "rtsp client disconnected: %s\n", ip);
    }

    static int onVencStream(void* pData, void* pArgs, void* pUserData) {
        auto* self = static_cast<Impl*>(pUserData);
        if (!self || !self->running.load()) {
            return CVI_SUCCESS;
        }

        auto* pstDataCtx = static_cast<APP_DATA_CTX_S*>(pArgs);
        if (!pstDataCtx) {
            return CVI_SUCCESS;
        }

        auto* pstVencChnCfg = static_cast<APP_VENC_CHN_CFG_S*>(pstDataCtx->stDataParam.pParam);
        if (!pstVencChnCfg) {
            return CVI_SUCCESS;
        }

        const int vencChn = static_cast<int>(pstVencChnCfg->VencChn);
        auto* pstStream = static_cast<VENC_STREAM_S*>(pData);
        if (!pstStream || pstStream->u32PackCount == 0) {
            return CVI_SUCCESS;
        }

        CVI_RTSP_CTX* ctx = nullptr;
        CVI_RTSP_SESSION* session = nullptr;
        {
            std::lock_guard<std::mutex> lk(self->mu);
            ctx = self->rtsp_ctx;
            auto it = self->sessions_by_venc_chn.find(vencChn);
            if (it != self->sessions_by_venc_chn.end()) {
                session = it->second;
            }
        }

        if (!ctx || !session) {
            return CVI_SUCCESS;
        }

        // NOTE: Do NOT touch RGN/OSD here.
        // VENC callback is latency-sensitive; heavy work may overflow LL cache and drop frames.

        CVI_RTSP_DATA data;
        std::memset(&data, 0, sizeof(CVI_RTSP_DATA));

        const CVI_U32 maxBlocks = static_cast<CVI_U32>(sizeof(data.dataPtr) / sizeof(data.dataPtr[0]));
        const CVI_U32 blockCnt = (pstStream->u32PackCount > maxBlocks) ? maxBlocks : pstStream->u32PackCount;
        data.blockCnt = blockCnt;

        for (CVI_U32 i = 0; i < blockCnt; i++) {
            VENC_PACK_S* ppack = &pstStream->pstPack[i];
            data.dataPtr[i] = ppack->pu8Addr + ppack->u32Offset;
            data.dataLen[i] = ppack->u32Len - ppack->u32Offset;
        }

        (void)CVI_RTSP_WriteFrame(ctx, session->video, &data);
        return CVI_SUCCESS;
    }

#if SESG_STREAM_RTSP_HAS_RGN
    static inline bool rgn_device_present() {
        // Different BSPs expose different device node names.
        // If none is present, CVI_RGN_* may spam kernel logs; fail fast.
        static const char* kNodes[] = {
            "/dev/cvi-rgn",
            "/dev/cvi_rgn",
            "/dev/rgn",
            "/dev/region",
            "/dev/cvi_region",
        };
        for (const char* p : kNodes) {
            if (::access(p, F_OK) == 0) return true;
        }
        return false;
    }

    template <typename T, typename = void>
    struct has_enPixelFmt : std::false_type {};
    template <typename T>
    struct has_enPixelFmt<T, std::void_t<decltype(std::declval<T&>().enPixelFmt)>> : std::true_type {};

    template <typename T, typename = void>
    struct has_enPixelFormat : std::false_type {};
    template <typename T>
    struct has_enPixelFormat<T, std::void_t<decltype(std::declval<T&>().enPixelFormat)>> : std::true_type {};

    template <typename T, typename = void>
    struct has_u32BgAlpha : std::false_type {};
    template <typename T>
    struct has_u32BgAlpha<T, std::void_t<decltype(std::declval<T&>().u32BgAlpha)>> : std::true_type {};

    template <typename T, typename = void>
    struct has_u32FgAlpha : std::false_type {};
    template <typename T>
    struct has_u32FgAlpha<T, std::void_t<decltype(std::declval<T&>().u32FgAlpha)>> : std::true_type {};

    template <typename T, typename = void>
    struct has_u8BgAlpha : std::false_type {};
    template <typename T>
    struct has_u8BgAlpha<T, std::void_t<decltype(std::declval<T&>().u8BgAlpha)>> : std::true_type {};

    template <typename T, typename = void>
    struct has_u8FgAlpha : std::false_type {};
    template <typename T>
    struct has_u8FgAlpha<T, std::void_t<decltype(std::declval<T&>().u8FgAlpha)>> : std::true_type {};

    template <typename OverlayAttrT>
    static inline void set_overlay_pixel_format(OverlayAttrT& o, PIXEL_FORMAT_E fmt) {
        if constexpr (has_enPixelFmt<OverlayAttrT>::value) {
            o.enPixelFmt = fmt;
        } else if constexpr (has_enPixelFormat<OverlayAttrT>::value) {
            o.enPixelFormat = fmt;
        }
    }

    template <typename OverlayChnAttrT>
    static inline void set_overlay_alpha_if_present(OverlayChnAttrT& o, uint32_t bg, uint32_t fg) {
        if constexpr (has_u32BgAlpha<OverlayChnAttrT>::value) {
            o.u32BgAlpha = bg;
        } else if constexpr (has_u8BgAlpha<OverlayChnAttrT>::value) {
            o.u8BgAlpha = static_cast<CVI_U8>(bg);
        }
        if constexpr (has_u32FgAlpha<OverlayChnAttrT>::value) {
            o.u32FgAlpha = fg;
        } else if constexpr (has_u8FgAlpha<OverlayChnAttrT>::value) {
            o.u8FgAlpha = static_cast<CVI_U8>(fg);
        }
    }

    bool ensureOverlayReady(int vencChn) {
        auto it = overlay_by_venc_chn.find(vencChn);
        if (it == overlay_by_venc_chn.end() || !it->second.enabled) {
            return false;
        }
        auto& st = it->second;
        if (st.rgn_ready) {
            return true;
        }

        // If RGN device node is missing in this rootfs, disable overlay to avoid kernel log spam.
        if (!rgn_device_present()) {
            st.enabled = false;
            return false;
        }

        const uint64_t nowMs = now_ms_steady();
        if (st.rgn_next_retry_ms != 0 && nowMs < st.rgn_next_retry_ms) {
            return false;
        }

        // Create overlay region sized to the channel output.
        if (st.w == 0 || st.h == 0) {
            // Fallback: try to get from channel config.
            for (const auto& c : channels) {
                if (static_cast<int>(c.ch) == vencChn) {
                    st.w = c.width;
                    st.h = c.height;
                    break;
                }
            }
        }
        if (st.w == 0 || st.h == 0) {
            return false;
        }

        st.canvas_argb1555.assign(static_cast<size_t>(st.w) * st.h, 0);

        MMF_CHN_S chn;
        std::memset(&chn, 0, sizeof(chn));
        chn.enModId = CVI_ID_VENC;
        chn.s32DevId = 0;
        chn.s32ChnId = vencChn;

        RGN_ATTR_S attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.enType = OVERLAY_RGN;
        set_overlay_pixel_format(attr.unAttr.stOverlay, PIXEL_FORMAT_ARGB_1555);
        attr.unAttr.stOverlay.stSize.u32Width = st.w;
        attr.unAttr.stOverlay.stSize.u32Height = st.h;
        attr.unAttr.stOverlay.u32BgColor = 0;

        RGN_CHN_ATTR_S chnAttr;
        std::memset(&chnAttr, 0, sizeof(chnAttr));
        chnAttr.bShow = CVI_TRUE;
        chnAttr.enType = OVERLAY_RGN;
        chnAttr.unChnAttr.stOverlayChn.stPoint.s32X = 0;
        chnAttr.unChnAttr.stOverlayChn.stPoint.s32Y = 0;
        set_overlay_alpha_if_present(chnAttr.unChnAttr.stOverlayChn, 0, 255);
        chnAttr.unChnAttr.stOverlayChn.u32Layer = 0;

        // Use a handle range less likely to conflict with other modules.
        // Keep attempts minimal; if driver is unavailable we must not storm ioctls.
        const int handle = 100 + vencChn;
        bool ok = false;
        if (CVI_RGN_Create(handle, &attr) == CVI_SUCCESS) {
            if (CVI_RGN_AttachToChn(handle, &chn, &chnAttr) == CVI_SUCCESS) {
                ok = true;
            } else {
                (void)CVI_RGN_Destroy(handle);
            }
        }

        if (!ok) {
            st.rgn_fail_count++;
            // Exponential-ish backoff: 200ms, 400ms, 800ms, ... capped at 5s.
            const uint64_t backoff = std::min<uint64_t>(5000, 200ull << std::min<uint32_t>(st.rgn_fail_count, 5));
            st.rgn_next_retry_ms = nowMs + backoff;
            // Auto-disable early to stop kernel spam and performance regression.
            // If you need re-try, restart the app after fixing BSP/RGN driver.
            st.enabled = false;
            return false;
        }

        st.rgn_handle = handle;
        st.rgn_ready = true;
        st.dirty = true;
        st.rgn_fail_count = 0;
        st.rgn_next_retry_ms = 0;
        return true;
    }

    void ensureOverlayReadyAndApply(int vencChn) {
        std::lock_guard<std::mutex> lk(mu);
        if (!ensureOverlayReady(vencChn)) {
            return;
        }
        auto& st = overlay_by_venc_chn[vencChn];
        if (!st.dirty) {
            return;
        }
        // Clear to transparent.
        std::fill(st.canvas_argb1555.begin(), st.canvas_argb1555.end(), 0);

        for (const auto& r : st.rects) {
            const float nx = clamp01(r.x);
            const float ny = clamp01(r.y);
            const float nw = clamp01(r.w);
            const float nh = clamp01(r.h);
            const int x1 = static_cast<int>(nx * st.w);
            const int y1 = static_cast<int>(ny * st.h);
            const int x2 = static_cast<int>((nx + nw) * st.w);
            const int y2 = static_cast<int>((ny + nh) * st.h);
            draw_rect_outline_argb1555(st.canvas_argb1555, st.w, st.h, x1, y1, x2, y2,
                                      argb8888_to_argb1555(r.argb), r.thickness);
        }

        BITMAP_S bmp;
        std::memset(&bmp, 0, sizeof(bmp));
        bmp.enPixelFormat = PIXEL_FORMAT_ARGB_1555;
        bmp.u32Width = st.w;
        bmp.u32Height = st.h;
        bmp.pData = reinterpret_cast<CVI_VOID*>(st.canvas_argb1555.data());
        (void)CVI_RGN_SetBitMap(st.rgn_handle, &bmp);
        st.dirty = false;
    }

    void destroyOverlaysLocked() {
        for (auto& kv : overlay_by_venc_chn) {
            auto& st = kv.second;
            if (!st.rgn_ready || st.rgn_handle < 0) continue;
            MMF_CHN_S chn;
            std::memset(&chn, 0, sizeof(chn));
            chn.enModId = CVI_ID_VENC;
            chn.s32DevId = 0;
            chn.s32ChnId = kv.first;
            (void)CVI_RGN_DetachFromChn(st.rgn_handle, &chn);
            (void)CVI_RGN_Destroy(st.rgn_handle);
            st.rgn_handle = -1;
            st.rgn_ready = false;
            st.canvas_argb1555.clear();
        }
    }
#endif

    bool createRtspServerAndSessions(const ServerConfig& server_cfg, const std::vector<ChannelConfig>& channels) {
        CVI_RTSP_CONFIG config;
        std::memset(&config, 0, sizeof(CVI_RTSP_CONFIG));
        config.port = server_cfg.port;

        if (CVI_RTSP_Create(&rtsp_ctx, &config) < 0) {
            rtsp_ctx = nullptr;
            return false;
        }
        if (CVI_RTSP_Start(rtsp_ctx) < 0) {
            CVI_RTSP_Destroy(&rtsp_ctx);
            rtsp_ctx = nullptr;
            return false;
        }

        CVI_RTSP_STATE_LISTENER listener;
        std::memset(&listener, 0, sizeof(CVI_RTSP_STATE_LISTENER));
        listener.onConnect = &onConnect;
        listener.argConn = rtsp_ctx;
        listener.onDisconnect = &onDisconnect;
        listener.argDisconn = rtsp_ctx;
        CVI_RTSP_SetListener(rtsp_ctx, &listener);

        for (const auto& ch_cfg : channels) {
            CVI_RTSP_SESSION_ATTR attr;
            std::memset(&attr, 0, sizeof(CVI_RTSP_SESSION_ATTR));
            attr.reuseFirstSource = 1;
            attr.video.codec = toRtspVideoCodec(ch_cfg.format);
            if (attr.video.codec == RTSP_VIDEO_NONE) {
                return false;
            }

            const std::string path = makeSessionPath(server_cfg, ch_cfg);
            std::snprintf(attr.name, sizeof(attr.name), "%s", path.c_str());

            CVI_RTSP_SESSION* session = nullptr;
            if (CVI_RTSP_CreateSession(rtsp_ctx, &attr, &session) < 0 || !session) {
                return false;
            }
            sessions_by_venc_chn[static_cast<int>(ch_cfg.ch)] = session;
        }

        return true;
    }

    void destroyRtspServerAndSessions() {
        std::lock_guard<std::mutex> lk(mu);

        if (!rtsp_ctx) {
            sessions_by_venc_chn.clear();
            return;
        }

        for (auto& kv : sessions_by_venc_chn) {
            if (kv.second) {
                CVI_RTSP_DestroySession(rtsp_ctx, kv.second);
            }
        }
        sessions_by_venc_chn.clear();

        CVI_RTSP_Stop(rtsp_ctx);
        CVI_RTSP_Destroy(&rtsp_ctx);
        rtsp_ctx = nullptr;
    }

    void unregisterHandlers() {
        if (consumer_index < 0 || channels.empty()) {
            return;
        }
        for (const auto& ch_cfg : channels) {
            (void)registerVideoFrameHandler(ch_cfg.ch, consumer_index, nullptr, nullptr);
        }
    }
};

SesgRtspVpssStreamer::SesgRtspVpssStreamer() : impl_(std::make_unique<Impl>()) {}

SesgRtspVpssStreamer::~SesgRtspVpssStreamer() {
    stop(false);
}

bool SesgRtspVpssStreamer::start(const ServerConfig& server_cfg, const std::vector<ChannelConfig>& channels,
                                 bool init_vpss) {
    if (!impl_) {
        return false;
    }
    if (impl_->running.load()) {
        return true;
    }
    if (server_cfg.port <= 0) {
        return false;
    }
    if (channels.empty()) {
        return false;
    }

    for (const auto& ch_cfg : channels) {
        if (ch_cfg.ch < VIDEO_CH0 || ch_cfg.ch >= VIDEO_CH_MAX) {
            return false;
        }
        if (toRtspVideoCodec(ch_cfg.format) == RTSP_VIDEO_NONE) {
            return false;
        }
    }

    if (init_vpss) {
#if defined(SESG_STREAM_RTSP_ENABLE_VPSS)
        if (sesg_vpss_init() != 0) {
            return false;
        }
        impl_->vpss_inited = true;
#else
        (void)init_vpss;
        return false;
#endif
    }

    if (initVideo() != 0) {
        stop(false);
        return false;
    }

    for (const auto& ch_cfg : channels) {
        video_ch_param_t param;
        param.format = ch_cfg.format;
        param.width = ch_cfg.width;
        param.height = ch_cfg.height;
        param.fps = ch_cfg.fps;
        if (setupVideo(ch_cfg.ch, &param) != 0) {
            stop(false);
            return false;
        }

        // 用 index=0 注册一份 consumer，写入 RTSP。
        if (registerVideoFrameHandler(ch_cfg.ch, 0, &Impl::onVencStream, impl_.get()) != 0) {
            stop(false);
            return false;
        }

        {
            std::lock_guard<std::mutex> lk(impl_->mu);
            auto& st = impl_->overlay_by_venc_chn[static_cast<int>(ch_cfg.ch)];
            st.enabled = ch_cfg.enable_rgn_overlay;
            st.w = ch_cfg.width;
            st.h = ch_cfg.height;
            st.dirty = true;
        }
    }

    if (!impl_->createRtspServerAndSessions(server_cfg, channels)) {
        // createRtspServerAndSessions 内部可能已经创建了部分资源
        impl_->destroyRtspServerAndSessions();
        stop(false);
        return false;
    }

    // 注意：components/sophgo/video 的 startVideo() 虽然声明返回 int，但当前实现没有 return，
    // 因此这里不做返回值判断。
    impl_->attached_mode = false;
    impl_->consumer_index = 0;
    impl_->channels = channels;

    startVideo();

#if SESG_STREAM_RTSP_HAS_RGN
    // Non-attached mode: system is started; eagerly create overlay regions.
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        for (const auto& ch_cfg : channels) {
            if (ch_cfg.enable_rgn_overlay) {
                (void)impl_->ensureOverlayReady(static_cast<int>(ch_cfg.ch));
            }
        }
    }
#endif

    impl_->running.store(true);
    return true;
}

bool SesgRtspVpssStreamer::startAttached(const ServerConfig& server_cfg, const std::vector<ChannelConfig>& channels,
                                        int consumer_index) {
    if (!impl_) {
        return false;
    }
    if (impl_->running.load()) {
        return true;
    }
    if (server_cfg.port <= 0) {
        return false;
    }
    if (channels.empty()) {
        return false;
    }
    if (consumer_index < 0) {
        return false;
    }

    for (const auto& ch_cfg : channels) {
        if (ch_cfg.ch < VIDEO_CH0 || ch_cfg.ch >= VIDEO_CH_MAX) {
            return false;
        }
        if (toRtspVideoCodec(ch_cfg.format) == RTSP_VIDEO_NONE) {
            return false;
        }
    }

    // 先记录配置，确保失败路径可以 stopAttached() 清理已注册 handler。
    impl_->attached_mode = true;
    impl_->consumer_index = consumer_index;
    impl_->channels = channels;

    // 附着模式不调用 initVideo/startVideo：避免覆盖外部模块已加载/调整的参数。
    for (const auto& ch_cfg : channels) {
        video_ch_param_t param;
        param.format = ch_cfg.format;
        param.width = ch_cfg.width;
        param.height = ch_cfg.height;
        param.fps = ch_cfg.fps;
        if (setupVideo(ch_cfg.ch, &param) != 0) {
            stopAttached();
            return false;
        }
        if (registerVideoFrameHandler(ch_cfg.ch, consumer_index, &Impl::onVencStream, impl_.get()) != 0) {
            stopAttached();
            return false;
        }

        {
            std::lock_guard<std::mutex> lk(impl_->mu);
            auto& st = impl_->overlay_by_venc_chn[static_cast<int>(ch_cfg.ch)];
            st.enabled = ch_cfg.enable_rgn_overlay;
            st.w = ch_cfg.width;
            st.h = ch_cfg.height;
            st.dirty = true;
            // attached mode: delay RGN init until VENC is running
        }
    }

    if (!impl_->createRtspServerAndSessions(server_cfg, channels)) {
        impl_->destroyRtspServerAndSessions();
        stopAttached();
        return false;
    }

    impl_->running.store(true);
    return true;
}

void SesgRtspVpssStreamer::stop(bool deinit_vpss) {
    if (!impl_) {
        return;
    }

    impl_->running.store(false);

    impl_->attached_mode = false;

    // 先停 video（停止 VENC 输出），再销毁 RTSP session。
    // 注意：deinitVideo() 同样存在“声明返回 int 但实现未 return”的情况，因此不做返回值判断。
    deinitVideo();

#if SESG_STREAM_RTSP_HAS_RGN
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->destroyOverlaysLocked();
    }
#endif

    impl_->destroyRtspServerAndSessions();

    if (deinit_vpss && impl_->vpss_inited) {
#if defined(SESG_STREAM_RTSP_ENABLE_VPSS)
        sesg_vpss_deinit();
#else
        (void)deinit_vpss;
#endif
        impl_->vpss_inited = false;
    }
}

void SesgRtspVpssStreamer::stopAttached() {
    if (!impl_) {
        return;
    }
    impl_->running.store(false);

    impl_->destroyRtspServerAndSessions();

#if SESG_STREAM_RTSP_HAS_RGN
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->destroyOverlaysLocked();
    }
#endif

    impl_->unregisterHandlers();

    impl_->attached_mode = false;
    impl_->consumer_index = 0;
    impl_->channels.clear();
}

bool SesgRtspVpssStreamer::isRunning() const {
    return impl_ && impl_->running.load();
}

bool SesgRtspVpssStreamer::updateOverlayRects(video_ch_index_t ch, const std::vector<OverlayRect>& rects) {
    if (!impl_) {
        return false;
    }
    const int vencChn = static_cast<int>(ch);
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        auto it = impl_->overlay_by_venc_chn.find(vencChn);
        if (it == impl_->overlay_by_venc_chn.end() || !it->second.enabled) {
            return false;
        }
        it->second.rects = rects;
        it->second.dirty = true;
    }

#if SESG_STREAM_RTSP_HAS_RGN
    // Best-effort immediate apply (if RGN already ready).
    impl_->ensureOverlayReadyAndApply(vencChn);
    return true;
#else
    return false;
#endif
}

} // namespace sesg::stream_rtsp
