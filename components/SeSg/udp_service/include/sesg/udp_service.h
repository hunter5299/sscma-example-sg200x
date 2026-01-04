#pragma once

#include <udp_service/udp_service.h>

namespace sesg {
namespace udp_service {

using PayloadFormat = ::udp_service::PayloadFormat;
using SenderOptions = ::udp_service::SenderOptions;

class SesgUDPSender : public ::udp_service::UDPSender {
public:
    using ::udp_service::UDPSender::UDPSender;
};

}  // namespace udp_service
}  // namespace sesg
