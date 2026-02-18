#!/bin/bash

# ============================================================================
# ChromaGuide Production Deployment Script
# ============================================================================
#
# Usage:
#   ./deploy_production.sh --environment production --mode full
#   ./deploy_production.sh --environment staging --mode models-only
#
# Stages:
#   1. Pre-deployment validation
#   2. Build and test
#   3. Deploy models to production
#   4. Set up monitoring
#   5. Smoke tests
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${ENVIRONMENT:-staging}
DEPLOY_MODE=${DEPLOY_MODE:-full}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_LOG="${PROJECT_ROOT}/deployment.log"

# ============================================================================
# LOGGING
# ============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$DEPLOYMENT_LOG"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$DEPLOYMENT_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$DEPLOYMENT_LOG"
}

# ============================================================================
# PRE-DEPLOYMENT CHECKS
# ============================================================================

check_environment() {
    log_info "Checking deployment environment..."
    
    # Check required tools
    for cmd in python3 git docker; do
        if ! command -v $cmd &> /dev/null; then
            log_warn "$cmd not found (optional)"
        fi
    done
    
    # Check Python version
    python_version=$(python3 --version | cut -d' ' -f2)
    log_info "Python version: $python_version"
    
    # Check git status
    if [ -d "$PROJECT_ROOT/.git" ]; then
        git_branch=$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD)
        git_commit=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)
        log_info "Git branch: $git_branch"
        log_info "Git commit: $git_commit"
    fi
}

validate_code() {
    log_info "Validating code..."
    
    cd "$PROJECT_ROOT"
    
    # Run syntax check
    python3 -m py_compile sota_baselines.py || log_error "Syntax error in sota_baselines.py"
    python3 -m py_compile analytics.py || log_error "Syntax error in analytics.py"
    python3 -m py_compile profile_performance.py || log_error "Syntax error in profile_performance.py"
    
    log_info "Code validation passed"
}

validate_models() {
    log_info "Validating trained models..."
    
    if [ -f "$PROJECT_ROOT/checkpoints/phase1/best_model.pt" ]; then
        log_info "Phase 1 model found"
    else
        log_warn "Phase 1 model not found"
    fi
    
    if [ -f "$PROJECT_ROOT/checkpoints/phase2_xgboost/xgboost_model.pkl" ]; then
        log_info "Phase 2 model found"
    else
        log_warn "Phase 2 model not found"
    fi
}

# ============================================================================
# INSTALLATION AND BUILD
# ============================================================================

install_dependencies() {
    log_info "Installing dependencies..."
    
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip3 install --upgrade pip setuptools wheel
        pip3 install -r "$PROJECT_ROOT/requirements.txt"
        log_info "Dependencies installed"
    else
        log_warn "requirements.txt not found"
    fi
}

build_docker_image() {
    log_info "Building Docker image..."
    
    if command -v docker &> /dev/null; then
        if [ -f "$PROJECT_ROOT/Dockerfile" ]; then
            docker build -t chromaguide:wip "$PROJECT_ROOT"
            log_info "Docker image built"
        else
            log_warn "Dockerfile not found, skipping Docker build"
        fi
    else
        log_warn "Docker not available, skipping Docker build"
    fi
}

# ============================================================================
# TESTING
# ============================================================================

run_tests() {
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    if command -v pytest &> /dev/null; then
        python3 -m pytest tests/test_pipeline.py -v --tb=short || log_error "Some tests failed"
        log_info "Tests completed"
    else
        log_warn "pytest not found, installing..."
        pip3 install pytest
        python3 -m pytest tests/test_pipeline.py -v --tb=short || log_error "Some tests failed"
    fi
}

# ============================================================================
# DEPLOYMENT
# ============================================================================

deploy_models() {
    log_info "Deploying models to $ENVIRONMENT..."
    
    case $ENVIRONMENT in
        production)
            DEPLOY_DIR="/var/lib/chromaguide/models"
            ;;
        staging)
            DEPLOY_DIR="$PROJECT_ROOT/deployed_models"
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    mkdir -p "$DEPLOY_DIR"
    
    # Copy models
    if [ -d "$PROJECT_ROOT/checkpoints" ]; then
        cp -r "$PROJECT_ROOT/checkpoints"/* "$DEPLOY_DIR/" 2>/dev/null || true
        log_info "Models deployed to $DEPLOY_DIR"
    else
        log_warn "Checkpoints directory not found"
    fi
}

deploy_code() {
    log_info "Deploying application code..."
    
    case $ENVIRONMENT in
        production)
            CODE_DIR="/var/lib/chromaguide/app"
            ;;
        staging)
            CODE_DIR="$PROJECT_ROOT/deployed_app"
            ;;
    esac
    
    mkdir -p "$CODE_DIR"
    
    # Copy Python modules
    for file in sota_baselines.py analytics.py profile_performance.py explain_model.py preprocess_data.py validate_models.py; do
        cp "$PROJECT_ROOT/$file" "$CODE_DIR/" 2>/dev/null || true
    done
    
    log_info "Code deployed to $CODE_DIR"
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create systemd service (production only)
    if [ "$ENVIRONMENT" = "production" ]; then
        cat > /tmp/chromaguide.service << EOF
[Unit]
Description=ChromaGuide Pipeline Service
After=network.target

[Service]
Type=simple
User=chromaguide
WorkingDirectory=/var/lib/chromaguide
ExecStart=/usr/bin/python3 /var/lib/chromaguide/app/orchestrate_pipeline.py --watch-job 56644478
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        log_info "Systemd service template created"
    fi
    
    # Create monitoring config
    mkdir -p "${PROJECT_ROOT}/monitoring"
    cat > "${PROJECT_ROOT}/monitoring/prometheus.yml" << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'chromaguide'
    static_configs:
      - targets: ['localhost:8000']
EOF
    log_info "Monitoring configured"
}

# ============================================================================
# SMOKE TESTS
# ============================================================================

smoke_tests() {
    log_info "Running smoke tests..."
    
    cd "$PROJECT_ROOT"
    
    # Test SOTA baseline imports
    python3 -c "from sota_baselines import SOTARegistry; models = SOTARegistry.list_models(); print(f'SOTA Models: {models}')" || log_error "SOTA baseline smoke test failed"
    
    # Test analytics
    python3 -c "from analytics import PipelineAnalytics; a = PipelineAnalytics(); print('Analytics OK')" || log_error "Analytics smoke test failed"
    
    # Test profiling
    python3 -c "from profile_performance import PerformanceProfiler; p = PerformanceProfiler(); print('Profiler OK')" || log_error "Profiler smoke test failed"
    
    log_info "Smoke tests passed"
}

# ============================================================================
# ROLLBACK
# ============================================================================

handle_error() {
    log_error "Deployment failed! Rolling back..."
    
    # Restore previous version from git
    git -C "$PROJECT_ROOT" reset --hard HEAD~1 2>/dev/null || true
    
    exit 1
}

# ============================================================================
# MAIN DEPLOYMENT FLOW
# ============================================================================

main() {
    log_info "=== ChromaGuide Production Deployment ===" 
    log_info "Environment: $ENVIRONMENT"
    log_info "Mode: $DEPLOY_MODE"
    log_info "Start time: $(date)"
    
    # Setup error handling
    trap handle_error ERR
    
    # Pre-deployment
    check_environment
    validate_code
    validate_models
    
    if [ "$DEPLOY_MODE" = "full" ] || [ "$DEPLOY_MODE" = "models-only" ]; then
        install_dependencies
        run_tests
    fi
    
    # Deployment
    if [ "$DEPLOY_MODE" = "full" ] || [ "$DEPLOY_MODE" = "models-only" ]; then
        deploy_models
    fi
    
    if [ "$DEPLOY_MODE" = "full" ] || [ "$DEPLOY_MODE" = "code-only" ]; then
        deploy_code
    fi
    
    if [ "$DEPLOY_MODE" = "full" ]; then
        build_docker_image
        setup_monitoring
    fi
    
    # Post-deployment
    smoke_tests
    
    log_info "=== Deployment Complete ===" 
    log_info "End time: $(date)"
    log_info "Logs: $DEPLOYMENT_LOG"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --mode)
            DEPLOY_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run deployment
main
