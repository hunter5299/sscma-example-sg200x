/**
 * @file onvif_service.cpp
 * @brief 最小可用 ONVIF 协议实现：WS-Discovery + Device/Media SOAP。
 *
 * 目标：让本设备在网络中表现为一台标准 ONVIF IP 摄像机（Profile S 子集），
 * 客户端（ONVIF Device Manager / onvif-zeep / VLC 等）可以：
 *   1) 通过 WS-Discovery 在局域网内自动发现本机；
 *   2) 通过 Device/Media SOAP 拿到设备信息、视频 profile 和 RTSP 流地址；
 *   3) 用返回的 rtsp://<ip>:<port>/<session> 拉取带 YOLO 检测框的 H.264 实时流。
 *
 * 实现说明：
 *   - WS-Discovery 走原生 POSIX UDP 组播（239.255.255.250:3702），无第三方依赖。
 *   - SOAP 服务复用仓库已有的 mongoose（7.17）做 HTTP 服务，按 SOAP Action 返回模板化 XML。
 *   - 只实现 demo 必需的核心接口，保持代码可读；非必需接口返回通用 Fault。
 */

#include "onvif_service.h"

#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "mongoose.h"

namespace onvif {

#define OTAG "onvif"

// ====================== 工具：字符串替换 ======================
static std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

// ====================== SOAP 响应模板 ======================
// 说明：占位符 {IP} {ONVIF_PORT} {RTSP_URI} {W} {H} {FPS} 等在运行时回填。

static const char* kSoapEnvOpen =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
    "<s:Envelope xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\" "
    "xmlns:tt=\"http://www.onvif.org/ver10/schema\" "
    "xmlns:tds=\"http://www.onvif.org/ver10/device/wsdl\" "
    "xmlns:trt=\"http://www.onvif.org/ver10/media/wsdl\">"
    "<s:Body>";
static const char* kSoapEnvClose = "</s:Body></s:Envelope>";

// GetDeviceInformation
static std::string buildDeviceInformation(const OnvifConfig& c) {
    std::string body =
        "<tds:GetDeviceInformationResponse>"
        "<tds:Manufacturer>{MANUF}</tds:Manufacturer>"
        "<tds:Model>{MODEL}</tds:Model>"
        "<tds:FirmwareVersion>{FW}</tds:FirmwareVersion>"
        "<tds:SerialNumber>{SN}</tds:SerialNumber>"
        "<tds:HardwareId>{HW}</tds:HardwareId>"
        "</tds:GetDeviceInformationResponse>";
    body = replace_all(body, "{MANUF}", c.info.manufacturer);
    body = replace_all(body, "{MODEL}", c.info.model);
    body = replace_all(body, "{FW}", c.info.firmware);
    body = replace_all(body, "{SN}", c.info.serial);
    body = replace_all(body, "{HW}", c.info.hardware);
    return std::string(kSoapEnvOpen) + body + kSoapEnvClose;
}

// GetCapabilities：回报 Device + Media 服务的 XAddr。
static std::string buildCapabilities(const OnvifConfig& c) {
    char xaddr[256];
    snprintf(xaddr, sizeof(xaddr), "http://%s:%d/onvif/device_service", c.device_ip.c_str(), c.onvif_port);
    char media[256];
    snprintf(media, sizeof(media), "http://%s:%d/onvif/media_service", c.device_ip.c_str(), c.onvif_port);
    std::string body =
        "<tds:GetCapabilitiesResponse><tds:Capabilities>"
        "<tt:Device><tt:XAddr>{DEV}</tt:XAddr></tt:Device>"
        "<tt:Media><tt:XAddr>{MEDIA}</tt:XAddr>"
        "<tt:StreamingCapabilities><tt:RTPMulticast>false</tt:RTPMulticast>"
        "<tt:RTP_TCP>true</tt:RTP_TCP><tt:RTP_RTSP_TCP>true</tt:RTP_RTSP_TCP>"
        "</tt:StreamingCapabilities></tt:Media>"
        "</tds:Capabilities></tds:GetCapabilitiesResponse>";
    body = replace_all(body, "{DEV}", xaddr);
    body = replace_all(body, "{MEDIA}", media);
    return std::string(kSoapEnvOpen) + body + kSoapEnvClose;
}

// GetServices：ONVIF 客户端常先调用此接口定位各服务 XAddr。
static std::string buildServices(const OnvifConfig& c) {
    char dev[256], media[256];
    snprintf(dev, sizeof(dev), "http://%s:%d/onvif/device_service", c.device_ip.c_str(), c.onvif_port);
    snprintf(media, sizeof(media), "http://%s:%d/onvif/media_service", c.device_ip.c_str(), c.onvif_port);
    std::string body =
        "<tds:GetServicesResponse>"
        "<tds:Service><tds:Namespace>http://www.onvif.org/ver10/device/wsdl</tds:Namespace>"
        "<tds:XAddr>{DEV}</tds:XAddr><tds:Version><tt:Major>2</tt:Major><tt:Minor>5</tt:Minor></tds:Version></tds:Service>"
        "<tds:Service><tds:Namespace>http://www.onvif.org/ver10/media/wsdl</tds:Namespace>"
        "<tds:XAddr>{MEDIA}</tds:XAddr><tds:Version><tt:Major>2</tt:Major><tt:Minor>5</tt:Minor></tds:Version></tds:Service>"
        "</tds:GetServicesResponse>";
    body = replace_all(body, "{DEV}", dev);
    body = replace_all(body, "{MEDIA}", media);
    return std::string(kSoapEnvOpen) + body + kSoapEnvClose;
}

// GetProfiles / GetProfile：描述单个视频 profile（编码、分辨率、帧率）。
static std::string profileXml(const OnvifConfig& c, bool single) {
    char buf[1024];
    snprintf(buf, sizeof(buf),
        "<trt:Profiles token=\"profile0\" fixed=\"true\">"
        "<tt:Name>YOLO_H264</tt:Name>"
        "<tt:VideoSourceConfiguration token=\"vsrc0\">"
        "<tt:Name>VideoSource</tt:Name><tt:UseCount>1</tt:UseCount>"
        "<tt:SourceToken>vsrctoken0</tt:SourceToken>"
        "<tt:Bounds x=\"0\" y=\"0\" width=\"%u\" height=\"%u\"/>"
        "</tt:VideoSourceConfiguration>"
        "<tt:VideoEncoderConfiguration token=\"venc0\">"
        "<tt:Name>H264</tt:Name><tt:UseCount>1</tt:UseCount>"
        "<tt:Encoding>H264</tt:Encoding>"
        "<tt:Resolution><tt:Width>%u</tt:Width><tt:Height>%u</tt:Height></tt:Resolution>"
        "<tt:Quality>5</tt:Quality>"
        "<tt:RateControl><tt:FrameRateLimit>%u</tt:FrameRateLimit>"
        "<tt:EncodingInterval>1</tt:EncodingInterval><tt:BitrateLimit>1500</tt:BitrateLimit></tt:RateControl>"
        "<tt:H264><tt:GovLength>25</tt:GovLength><tt:H264Profile>Main</tt:H264Profile></tt:H264>"
        "</tt:VideoEncoderConfiguration>"
        "</trt:Profiles>",
        c.width, c.height, c.width, c.height, c.fps);
    std::string one = buf;
    // GetProfiles 返回数组形式（这里只有一个），GetProfile 返回单个，标签名一致即可。
    (void)single;
    return one;
}

static std::string buildProfiles(const OnvifConfig& c) {
    std::string body = "<trt:GetProfilesResponse>" + profileXml(c, false) + "</trt:GetProfilesResponse>";
    return std::string(kSoapEnvOpen) + body + kSoapEnvClose;
}

// GetStreamUri：返回 RTSP 拉流地址，这是 demo 的关键——把内部 RTSP 引擎地址告诉客户端。
static std::string buildStreamUri(const OnvifConfig& c) {
    char uri[256];
    snprintf(uri, sizeof(uri), "rtsp://%s:%d/%s", c.device_ip.c_str(), c.rtsp_port, c.rtsp_session.c_str());
    std::string body =
        "<trt:GetStreamUriResponse><trt:MediaUri>"
        "<tt:Uri>{URI}</tt:Uri>"
        "<tt:InvalidAfterConnect>false</tt:InvalidAfterConnect>"
        "<tt:InvalidAfterReboot>false</tt:InvalidAfterReboot>"
        "<tt:Timeout>PT0S</tt:Timeout>"
        "</trt:MediaUri></trt:GetStreamUriResponse>";
    body = replace_all(body, "{URI}", uri);
    return std::string(kSoapEnvOpen) + body + kSoapEnvClose;
}

// 通用 SOAP Fault（未实现的接口）。
static std::string buildFault(const char* reason) {
    std::string body =
        "<s:Fault><s:Code><s:Value>s:Receiver</s:Value></s:Code>"
        "<s:Reason><s:Text xml:lang=\"en\">";
    body += reason;
    body += "</s:Text></s:Reason></s:Fault>";
    return std::string(kSoapEnvOpen) + body + kSoapEnvClose;
}

// ====================== mongoose HTTP / SOAP 处理 ======================

// 根据请求体里出现的 SOAP Action 关键字选择响应。
// ONVIF 客户端的 action 既可能在 HTTP SOAPAction 头，也可能在 body 的元素名里，
// 这里统一在 body 文本里匹配元素名，最稳妥。
static std::string dispatchSoap(const OnvifConfig& c, const std::string& body) {
    auto has = [&](const char* kw) { return body.find(kw) != std::string::npos; };

    if (has("GetDeviceInformation")) return buildDeviceInformation(c);
    if (has("GetCapabilities"))      return buildCapabilities(c);
    if (has("GetServices"))          return buildServices(c);
    if (has("GetStreamUri"))         return buildStreamUri(c);
    if (has("GetProfiles"))          return buildProfiles(c);
    if (has("GetProfile"))           return buildProfiles(c);
    if (has("GetVideoSources"))      return buildProfiles(c); // 简化：复用 profile 描述
    if (has("GetSystemDateAndTime")) return buildDeviceInformation(c); // 客户端探活，给个合法响应即可
    return buildFault("ActionNotSupported");
}

static void http_handler(struct mg_connection* conn, int ev, void* ev_data) {
    if (ev != MG_EV_HTTP_MSG) return;
    auto* hm = static_cast<struct mg_http_message*>(ev_data);
    auto* cfg = static_cast<OnvifConfig*>(conn->fn_data);

    std::string body(hm->body.buf, hm->body.len);
    std::string resp = dispatchSoap(*cfg, body);

    mg_http_reply(conn, 200,
                  "Content-Type: application/soap+xml; charset=utf-8\r\n",
                  "%s", resp.c_str());
}

void OnvifService::httpLoop() {
    struct mg_mgr mgr;
    mg_mgr_init(&mgr);
    char url[64];
    snprintf(url, sizeof(url), "http://0.0.0.0:%d", cfg_.onvif_port);
    if (!mg_http_listen(&mgr, url, http_handler, &cfg_)) {
        printf("[%s] ERROR: mg_http_listen failed on %s\n", OTAG, url);
        mg_mgr_free(&mgr);
        return;
    }
    printf("[%s] SOAP HTTP service on %s (device_service / media_service)\n", OTAG, url);
    while (running_.load()) {
        mg_mgr_poll(&mgr, 200);
    }
    mg_mgr_free(&mgr);
}

// ====================== WS-Discovery (UDP 组播 3702) ======================

static const char* kWsdAddr = "239.255.255.250";
static const int   kWsdPort = 3702;

// 从 Probe 请求里提取 MessageID，用作 ProbeMatch 的 RelatesTo（不强制，提取失败给默认值）。
static std::string extractMessageId(const std::string& xml) {
    // 粗匹配 <wsa:MessageID>...</...> 或 <MessageID>...</...>
    static const char* tags[] = {"MessageID>", "messageid>"};
    for (const char* t : tags) {
        size_t p = xml.find(t);
        if (p == std::string::npos) continue;
        p += strlen(t);
        size_t e = xml.find('<', p);
        if (e == std::string::npos) continue;
        return xml.substr(p, e - p);
    }
    return "uuid:00000000-0000-0000-0000-000000000000";
}

// 构造 ProbeMatch 响应：声明本机为 NetworkVideoTransmitter，并给出 Device 服务 XAddr。
static std::string buildProbeMatch(const OnvifConfig& c, const std::string& relates_to) {
    char xaddr[256];
    snprintf(xaddr, sizeof(xaddr), "http://%s:%d/onvif/device_service", c.device_ip.c_str(), c.onvif_port);

    std::string tmpl =
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<s:Envelope xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\" "
        "xmlns:a=\"http://schemas.xmlsoap.org/ws/2004/08/addressing\" "
        "xmlns:d=\"http://schemas.xmlsoap.org/ws/2005/04/discovery\" "
        "xmlns:dn=\"http://www.onvif.org/ver10/network/wsdl\" "
        "xmlns:tds=\"http://www.onvif.org/ver10/device/wsdl\">"
        "<s:Header>"
        "<a:MessageID>uuid:{MSGID}</a:MessageID>"
        "<a:RelatesTo>{RELATES}</a:RelatesTo>"
        "<a:To>http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous</a:To>"
        "<a:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/ProbeMatches</a:Action>"
        "</s:Header><s:Body>"
        "<d:ProbeMatches><d:ProbeMatch>"
        "<a:EndpointReference><a:Address>urn:uuid:{SN}</a:Address></a:EndpointReference>"
        "<d:Types>dn:NetworkVideoTransmitter tds:Device</d:Types>"
        "<d:Scopes>"
        "onvif://www.onvif.org/type/video_encoder "
        "onvif://www.onvif.org/Profile/Streaming "
        "onvif://www.onvif.org/name/{MODEL} "
        "onvif://www.onvif.org/hardware/{HW}"
        "</d:Scopes>"
        "<d:XAddrs>{XADDR}</d:XAddrs>"
        "<d:MetadataVersion>1</d:MetadataVersion>"
        "</d:ProbeMatch></d:ProbeMatches>"
        "</s:Body></s:Envelope>";

    tmpl = replace_all(tmpl, "{MSGID}", c.info.serial);
    tmpl = replace_all(tmpl, "{RELATES}", relates_to);
    tmpl = replace_all(tmpl, "{SN}", c.info.serial);
    tmpl = replace_all(tmpl, "{MODEL}", c.info.model);
    tmpl = replace_all(tmpl, "{HW}", c.info.hardware);
    tmpl = replace_all(tmpl, "{XADDR}", xaddr);
    return tmpl;
}

void OnvifService::discoveryLoop() {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        printf("[%s] ERROR: discovery socket() failed\n", OTAG);
        return;
    }
    disc_fd_ = fd;

    int reuse = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(kWsdPort);
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        printf("[%s] ERROR: discovery bind(:%d) failed\n", OTAG, kWsdPort);
        close(fd);
        disc_fd_ = -1;
        return;
    }

    // 加入组播组 239.255.255.250。
    struct ip_mreq mreq;
    memset(&mreq, 0, sizeof(mreq));
    mreq.imr_multiaddr.s_addr = inet_addr(kWsdAddr);
    mreq.imr_interface.s_addr = htonl(INADDR_ANY);
    if (setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        printf("[%s] WARN: IP_ADD_MEMBERSHIP failed (discovery may not work)\n", OTAG);
    }

    // 接收超时，便于优雅退出。
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 300 * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    printf("[%s] WS-Discovery listening on %s:%d\n", OTAG, kWsdAddr, kWsdPort);

    char buf[4096];
    while (running_.load()) {
        struct sockaddr_in src;
        socklen_t slen = sizeof(src);
        ssize_t n = recvfrom(fd, buf, sizeof(buf) - 1, 0, (struct sockaddr*)&src, &slen);
        if (n <= 0) continue;  // 超时或错误
        buf[n] = '\0';
        std::string req(buf, n);

        // 只响应 Probe 请求。
        if (req.find("Probe") == std::string::npos) continue;

        std::string relates = extractMessageId(req);
        std::string resp = buildProbeMatch(cfg_, relates);

        // 单播回送到请求来源（WS-Discovery 标准做法）。
        sendto(fd, resp.data(), resp.size(), 0, (struct sockaddr*)&src, slen);
        printf("[%s] Probe from %s -> ProbeMatch sent (%zu bytes)\n",
               OTAG, inet_ntoa(src.sin_addr), resp.size());
    }

    close(fd);
    disc_fd_ = -1;
}

// ====================== 生命周期 ======================
OnvifService::OnvifService() = default;
OnvifService::~OnvifService() { stop(); }

bool OnvifService::start(const OnvifConfig& cfg) {
    if (running_.load()) return true;
    cfg_ = cfg;
    if (cfg_.device_ip.empty()) {
        printf("[%s] ERROR: device_ip is empty\n", OTAG);
        return false;
    }
    running_.store(true);
    http_thread_ = std::thread(&OnvifService::httpLoop, this);
    disc_thread_ = std::thread(&OnvifService::discoveryLoop, this);
    printf("[%s] started: ip=%s onvif_port=%d rtsp=rtsp://%s:%d/%s\n",
           OTAG, cfg_.device_ip.c_str(), cfg_.onvif_port,
           cfg_.device_ip.c_str(), cfg_.rtsp_port, cfg_.rtsp_session.c_str());
    return true;
}

void OnvifService::stop() {
    if (!running_.load()) return;
    running_.store(false);
    if (disc_fd_ >= 0) {
        // 触发 recvfrom 退出靠超时；这里不强制 close 以避免线程内重复 close。
    }
    if (http_thread_.joinable()) http_thread_.join();
    if (disc_thread_.joinable()) disc_thread_.join();
    printf("[%s] stopped\n", OTAG);
}

} // namespace onvif
