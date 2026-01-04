#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Thin wrapper around sophgo/video VPSS init/deinit.
// Keep the public header independent from sophgo headers to avoid include-order issues.
int sesg_vpss_init(void);
int sesg_vpss_deinit(void);

#ifdef __cplusplus
}
#endif
