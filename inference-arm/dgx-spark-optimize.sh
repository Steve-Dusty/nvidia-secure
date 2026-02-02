#!/bin/bash
#===============================================================================
# NVIDIA DGX Spark ARM Architecture Optimization Script
# Optimizes GPU-CPU memory coherency for NVIDIA models on Grace-Blackwell
#===============================================================================

set -euo pipefail

# Configuration
LOG_FILE="/var/log/dgx-spark-optimize.log"
NUMA_BALANCING_ENABLED=1
HUGEPAGES_SIZE="1G"
HUGEPAGES_COUNT=64

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "This script must be run as root"
        exit 1
    fi
}

check_dgx_spark() {
    log "Verifying NVIDIA DGX Spark platform..."

    # Check for Grace CPU (ARM Neoverse V2)
    if ! grep -q "Neoverse-V2\|grace" /proc/cpuinfo 2>/dev/null; then
        log "WARNING: Grace CPU not detected. Proceeding with generic ARM optimizations."
    fi

    # Check for Blackwell GPU
    if ! nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi "blackwell\|B200\|GB200"; then
        log "WARNING: Blackwell GPU not detected. Some optimizations may not apply."
    fi

    log "Platform verification complete."
}

#===============================================================================
# ARM Architecture Optimizations
#===============================================================================

configure_arm_cpu() {
    log "Configuring ARM Grace CPU optimizations..."

    # Set CPU governor to performance mode
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [[ -f "$cpu" ]]; then
            echo "performance" > "$cpu"
        fi
    done

    # Disable CPU idle states for low latency (optional - increases power)
    for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
        if [[ -f "$state" ]]; then
            echo 1 > "$state"
        fi
    done

    # Configure ARM-specific prefetcher settings
    if [[ -f /sys/devices/system/cpu/cpu0/cache/index0/prefetch_control ]]; then
        for prefetch in /sys/devices/system/cpu/cpu*/cache/index*/prefetch_control; do
            echo 1 > "$prefetch" 2>/dev/null || true
        done
    fi

    log "ARM CPU configuration complete."
}

#===============================================================================
# NVLink-C2C Memory Coherency Configuration
#===============================================================================

configure_nvlink_c2c() {
    log "Configuring NVLink-C2C for GPU-CPU memory coherency..."

    # Enable coherent memory access between Grace and Blackwell
    if [[ -f /sys/module/nvidia/parameters/NVreg_EnableGpuFirmware ]]; then
        echo 1 > /sys/module/nvidia/parameters/NVreg_EnableGpuFirmware
    fi

    # Configure unified memory preferences
    nvidia-smi -pm 1  # Enable persistence mode
    nvidia-smi --auto-boost-default=0  # Disable auto-boost for consistent performance

    # Set memory clock to maximum
    nvidia-smi -lgc $(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1) 2>/dev/null || true

    # Enable GPU Direct RDMA for peer access
    if [[ -d /sys/kernel/mm/memory_peers ]]; then
        for peer in /sys/kernel/mm/memory_peers/*/; do
            echo 1 > "${peer}enabled" 2>/dev/null || true
        done
    fi

    log "NVLink-C2C configuration complete."
}

#===============================================================================
# Unified Memory and NUMA Configuration
#===============================================================================

configure_unified_memory() {
    log "Configuring unified memory for Grace-Blackwell architecture..."

    # Enable NUMA balancing for optimal memory placement
    echo $NUMA_BALANCING_ENABLED > /proc/sys/kernel/numa_balancing

    # Configure transparent hugepages for large model weights
    echo "always" > /sys/kernel/mm/transparent_hugepage/enabled
    echo "defer+madvise" > /sys/kernel/mm/transparent_hugepage/defrag

    # Allocate 1GB hugepages for GPU memory mappings
    echo $HUGEPAGES_COUNT > /sys/kernel/mm/hugepages/hugepages-${HUGEPAGES_SIZE}/nr_hugepages

    # Mount hugetlbfs if not mounted
    if ! mount | grep -q hugetlbfs; then
        mkdir -p /mnt/hugepages
        mount -t hugetlbfs -o pagesize=${HUGEPAGES_SIZE} none /mnt/hugepages
    fi

    # Set vm.zone_reclaim_mode for NUMA optimization
    sysctl -w vm.zone_reclaim_mode=0

    # Optimize memory overcommit for large models
    sysctl -w vm.overcommit_memory=1
    sysctl -w vm.overcommit_ratio=95

    # Increase max memory map areas for large tensor allocations
    sysctl -w vm.max_map_count=2097152

    log "Unified memory configuration complete."
}

#===============================================================================
# GPU Memory Configuration
#===============================================================================

configure_gpu_memory() {
    log "Configuring GPU memory optimizations..."

    # Enable MIG mode if supported (for partitioning)
    # Uncomment if MIG partitioning is needed:
    # nvidia-smi -mig 1

    # Configure GPU memory growth strategy via environment
    cat >> /etc/environment << 'EOF'
# NVIDIA GPU Memory Configuration
CUDA_VISIBLE_DEVICES=0
TF_FORCE_GPU_ALLOW_GROWTH=true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
CUDA_DEVICE_ORDER=PCI_BUS_ID
EOF

    # Configure NCCL for optimal multi-GPU communication
    cat >> /etc/environment << 'EOF'
# NCCL Configuration for Grace-Blackwell
NCCL_DEBUG=INFO
NCCL_IB_DISABLE=0
NCCL_NET_GDR_LEVEL=5
NCCL_P2P_LEVEL=NVL
EOF

    log "GPU memory configuration complete."
}

#===============================================================================
# Kernel Parameters for AI Workloads
#===============================================================================

configure_kernel_params() {
    log "Configuring kernel parameters for AI workloads..."

    cat > /etc/sysctl.d/99-dgx-spark-optimize.conf << 'EOF'
# Network optimizations for distributed training
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 250000
net.ipv4.tcp_no_metrics_save = 1
net.ipv4.tcp_congestion_control = bbr

# Memory optimizations
vm.dirty_ratio = 40
vm.dirty_background_ratio = 10
vm.swappiness = 10
vm.vfs_cache_pressure = 50

# File descriptor limits for large models
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288

# Shared memory for IPC between CPU and GPU processes
kernel.shmmax = 68719476736
kernel.shmall = 4294967296

# Real-time scheduling for GPU kernels
kernel.sched_rt_runtime_us = -1
EOF

    sysctl -p /etc/sysctl.d/99-dgx-spark-optimize.conf

    log "Kernel parameters configured."
}

#===============================================================================
# CUDA and cuDNN Optimization
#===============================================================================

configure_cuda_optimization() {
    log "Configuring CUDA runtime optimizations..."

    # Create optimization profile
    mkdir -p /etc/nvidia
    cat > /etc/nvidia/cuda-optimize.conf << 'EOF'
# CUDA Runtime Optimizations for DGX Spark

# Enable tensor cores by default
CUDA_TENSOR_CORE_MATH=1

# Optimize for Blackwell architecture (SM 100)
CUDA_ARCH_LIST=100

# Enable unified memory prefetching
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

# cuDNN autotuning for optimal kernel selection
CUDNN_BENCHMARK=1

# Enable Flash Attention optimizations
FLASH_ATTENTION_FORCE_BUILD=TRUE

# Memory pool configuration
CUDA_MEMPOOL_ENABLE=1
EOF

    # Source in profile
    cat >> /etc/profile.d/cuda-optimize.sh << 'EOF'
#!/bin/bash
source /etc/nvidia/cuda-optimize.conf
export CUDA_TENSOR_CORE_MATH
export CUDA_ARCH_LIST
export CUDA_MANAGED_FORCE_DEVICE_ALLOC
export CUDNN_BENCHMARK
export FLASH_ATTENTION_FORCE_BUILD
export CUDA_MEMPOOL_ENABLE
EOF

    chmod +x /etc/profile.d/cuda-optimize.sh

    log "CUDA optimization configured."
}

#===============================================================================
# Power and Thermal Management
#===============================================================================

configure_power_management() {
    log "Configuring power and thermal management..."

    # Set GPU power limit to maximum for performance
    MAX_POWER=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits | head -1)
    nvidia-smi -pl $MAX_POWER 2>/dev/null || true

    # Configure fan curve for sustained workloads (if controllable)
    nvidia-smi -gtt 85 2>/dev/null || true  # Set GPU target temperature

    # Disable power saving on PCIe
    for pcie in /sys/bus/pci/devices/*/power/control; do
        echo "on" > "$pcie" 2>/dev/null || true
    done

    log "Power management configured."
}

#===============================================================================
# Verification and Status
#===============================================================================

verify_configuration() {
    log "Verifying configuration..."

    echo ""
    echo "=========================================="
    echo "DGX Spark Optimization Status"
    echo "=========================================="
    echo ""

    echo "CPU Configuration:"
    echo "  Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'N/A')"
    echo "  Cores: $(nproc)"
    echo ""

    echo "Memory Configuration:"
    echo "  Total: $(free -h | awk '/^Mem:/{print $2}')"
    echo "  Hugepages (1G): $(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages 2>/dev/null || echo 'N/A')"
    echo "  NUMA Balancing: $(cat /proc/sys/kernel/numa_balancing)"
    echo ""

    echo "GPU Configuration:"
    nvidia-smi --query-gpu=name,memory.total,power.limit,clocks.current.memory --format=csv
    echo ""

    echo "NVLink Status:"
    nvidia-smi nvlink --status 2>/dev/null || echo "  NVLink status not available"
    echo ""

    log "Configuration verified."
}

#===============================================================================
# Rollback Function
#===============================================================================

rollback() {
    log "Rolling back optimizations..."

    # Reset CPU governor
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo "schedutil" > "$cpu" 2>/dev/null || true
    done

    # Reset NUMA balancing
    echo 1 > /proc/sys/kernel/numa_balancing

    # Reset GPU settings
    nvidia-smi -pm 0
    nvidia-smi --auto-boost-default=1
    nvidia-smi -rgc

    # Remove custom sysctl
    rm -f /etc/sysctl.d/99-dgx-spark-optimize.conf
    sysctl --system

    log "Rollback complete."
}

#===============================================================================
# Main Execution
#===============================================================================

main() {
    log "Starting DGX Spark ARM Architecture Optimization..."

    check_root
    check_dgx_spark

    case "${1:-apply}" in
        apply)
            configure_arm_cpu
            configure_nvlink_c2c
            configure_unified_memory
            configure_gpu_memory
            configure_kernel_params
            configure_cuda_optimization
            configure_power_management
            verify_configuration
            log "All optimizations applied successfully."
            ;;
        verify)
            verify_configuration
            ;;
        rollback)
            rollback
            ;;
        *)
            echo "Usage: $0 {apply|verify|rollback}"
            exit 1
            ;;
    esac
}

main "$@"
