#include <sesg/vpss.h>

// Provided by components/sophgo/video
extern int app_ipcam_Vpss_Init(void);
extern int app_ipcam_Vpss_DeInit(void);

int sesg_vpss_init(void) {
    return app_ipcam_Vpss_Init();
}

int sesg_vpss_deinit(void) {
    return app_ipcam_Vpss_DeInit();
}
