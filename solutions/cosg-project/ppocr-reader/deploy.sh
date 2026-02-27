#!/bin/bash
# deploy.sh - Deploy ppocr-reader to reCamera device
# Handles: stop all camera-using services → install deb → start service → verify
#
# Usage:
#   ./deploy.sh                          # Build + deploy
#   ./deploy.sh --skip-build             # Deploy existing deb only
#   ./deploy.sh --host 10.0.0.1          # Custom host
#   ./deploy.sh --password mypass        # Custom password

set -e

# --- Defaults ---
HOST="${RECAMERA_HOST:-192.168.42.1}"
USER="${RECAMERA_USER:-recamera}"
PASS="${RECAMERA_PASS:-recamera.2}"
SKIP_BUILD=false
MQTT_CHECK=true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOLUTION_NAME="ppocr-reader"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)       HOST="$2"; shift 2 ;;
        --user)       USER="$2"; shift 2 ;;
        --password)   PASS="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --no-mqtt)    MQTT_CHECK=false; shift ;;
        -h|--help)
            echo "Usage: $0 [--host IP] [--user USER] [--password PASS] [--skip-build] [--no-mqtt]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log()  { echo "==> $*"; }
warn() { echo "WARN: $*"; }
err()  { echo "ERROR: $*" >&2; exit 1; }
ok()   { echo "OK: $*"; }

# SSH/SCP wrappers - avoid eval and quoting issues
run_ssh() {
    sshpass -p "$PASS" ssh $SSH_OPTS "${USER}@${HOST}" "$@"
}

run_scp() {
    sshpass -p "$PASS" scp $SSH_OPTS "$@"
}

run_sudo() {
    # Run a command with sudo on the device
    run_ssh "printf '%s\n' '${PASS}' | sudo -S $*"
}

# --- Pre-flight ---
command -v sshpass >/dev/null || err "sshpass not found. Install: brew install hudochenkov/sshpass/sshpass"
ping -c 1 -t 2 "$HOST" >/dev/null 2>&1 || err "Device $HOST not reachable"

# --- Step 1: Build ---
if [ "$SKIP_BUILD" = false ]; then
    log "Building ${SOLUTION_NAME}..."
    docker exec ubuntu_dev_x86 bash -c "
        export SG200X_SDK_PATH=/workspace/sg2002_recamera_emmc
        export PATH=/workspace/host-tools/gcc/riscv64-linux-musl-x86_64/bin:\$PATH
        cd /workspace/sscma-example-sg200x/solutions/${SOLUTION_NAME}
        rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Release . && cmake --build build -j4 && cd build && cpack
    " || err "Build failed"
    ok "Build succeeded"
fi

# Find deb package
DEB_FILE=$(ls -t "${SCRIPT_DIR}/build/${SOLUTION_NAME}_"*_riscv64.deb 2>/dev/null | head -1)
[ -f "$DEB_FILE" ] || err "No deb package found in ${SCRIPT_DIR}/build/"
DEB_NAME=$(basename "$DEB_FILE")
log "Package: $DEB_NAME"

# --- Step 2: Stop ALL camera-using services on device ---
log "Stopping all camera-using services on ${HOST}..."

run_ssh 'for svc in /etc/init.d/S*sscma-node* /etc/init.d/K*sscma-node* \
               /etc/init.d/S*node-red* /etc/init.d/K*node-red* \
               /etc/init.d/S*sscma-supervisor* /etc/init.d/K*sscma-supervisor* \
               /etc/init.d/S*yolo*detector* /etc/init.d/K*yolo*detector* \
               /etc/init.d/S*ppocr* /etc/init.d/K*ppocr* \
               /etc/init.d/S*face-analysis* /etc/init.d/K*face-analysis*; do
    [ -x "$svc" ] && "$svc" stop 2>/dev/null || true
done' || warn "Some init scripts not found (OK)"

run_sudo 'killall -q ppocr-reader yolo8-detector yolo11-detector yolo26-detector yolo11s-detector face-analysis sscma-node 2>/dev/null || true'
sleep 2
ok "Services stopped"

# --- Step 3: Transfer & Install ---
log "Uploading ${DEB_NAME}..."
run_scp "$DEB_FILE" "${USER}@${HOST}:/tmp/" || err "SCP failed"
ok "Upload complete"

log "Installing package..."
run_sudo "opkg install --force-reinstall /tmp/${DEB_NAME}" || err "Install failed"
ok "Package installed"

# --- Step 4: Start service ---
log "Starting ${SOLUTION_NAME}..."
run_sudo "/etc/init.d/S92ppocr-reader restart" || err "Service start failed"
ok "Service started"

# --- Step 5: Verify ---
sleep 5

log "Checking service status..."
run_ssh "/etc/init.d/S92ppocr-reader status" || warn "Status check failed"

if [ "$MQTT_CHECK" = true ]; then
    log "Capturing MQTT output (10s, max 3 messages)..."
    run_ssh "timeout 10 mosquitto_sub -h localhost -t 'recamera/ppocr/texts' -C 3" || warn "MQTT check timed out"
fi

echo ""
ok "Deploy complete: ${DEB_NAME} -> ${HOST}"
