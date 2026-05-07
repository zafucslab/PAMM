#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-3}"
RUNNER_SCRIPT="${REPO_ROOT}/test/run_representative_multivariate.py"
SPLITS="${SPLITS:-M}"
PHASE="${PHASE:-both}"  # tuning | eval | both
DATASETS="${DATASETS:-SMD}"
# U NAB WSD Stock MGAB TAO UCR YAHOO CATSv2 Daphnet Exathlon IOPS LTDB MGAB MITDB NEK OPPORTUNITY Power SED SVDB SWaT TODS
#DATASETS="${DATASETS:-MSL SMAP SMD CATSv2 SWaT CreditCard Daphnet Exathlon GECCO Genesis GHL
#LTDB MITDB OPPORTUNITY PSM SVDB}" # M
MODELS="${MODELS:-PAMM}"
OVERWRITE="${OVERWRITE:-1}"
NO_SAVE_SCORES="${NO_SAVE_SCORES:-0}"
PAMM_WIN_SIZE="${PAMM_WIN_SIZE:-64}"
PAMM_LR="${PAMM_LR:-0.0001}"
PAMM_BATCH_SIZE="${PAMM_BATCH_SIZE:-64}"
PAMM_EPOCHS="${PAMM_EPOCHS:-20}"
PAMM_D_MODEL="${PAMM_D_MODEL:-128}"
PAMM_PATCH_SIZE="${PAMM_PATCH_SIZE:-16}"
PAMM_PATCH_STRIDE="${PAMM_PATCH_STRIDE:-2}"
PAMM_USE_LOCALITY_BIAS="${PAMM_USE_LOCALITY_BIAS:-1}"
PAMM_CONTRAST_WEIGHT="${PAMM_CONTRAST_WEIGHT:-0.0}"
PAMM_USE_REVIN="${PAMM_USE_REVIN:-1}"
POINT_AGGREGATE_MODE="${POINT_AGGREGATE_MODE:-center_weighted}"
POINT_CENTER_POWER="${POINT_CENTER_POWER:-1}"
POINT_GAUSSIAN_SIGMA="${POINT_GAUSSIAN_SIGMA:-1}"
SCORE_PROJECTION_MODE="${SCORE_PROJECTION_MODE:-mean}"
SCORE_PROJECTION_CENTER_POWER="${SCORE_PROJECTION_CENTER_POWER:-1.5}"
DIFF_PREJUDGE_ENABLED="${DIFF_PREJUDGE_ENABLED:-0}"
DIFF_PREJUDGE_QUANTILE="${DIFF_PREJUDGE_QUANTILE:-1}"
DIFF_PREJUDGE_COSINE_QUANTILE="${DIFF_PREJUDGE_COSINE_QUANTILE:-0.05}"
DIFF_PREJUDGE_MARGIN="${DIFF_PREJUDGE_MARGIN:-1.0}"
DIFF_PREJUDGE_SUPPRESSION="${DIFF_PREJUDGE_SUPPRESSION:-0.5}"
PATCH_CHANNEL_TOPK_WEIGHT="${PATCH_CHANNEL_TOPK_WEIGHT:-0.2}"
PATCH_CHANNEL_TOPK_RATIO="${PATCH_CHANNEL_TOPK_RATIO:-0.2}"
CNN_PATTERN_ENABLED="${CNN_PATTERN_ENABLED:-1}"
CNN_PATTERN_SCORE_WEIGHT="${CNN_PATTERN_SCORE_WEIGHT:-0.5}"
CNN_PATTERN_HIDDEN_DIM="${CNN_PATTERN_HIDDEN_DIM:-32}"
CNN_PATTERN_EMBEDDING_DIM="${CNN_PATTERN_EMBEDDING_DIM:-16}"
CNN_PATTERN_NUM_CONTEXTS="${CNN_PATTERN_NUM_CONTEXTS:-10}"
CNN_PATTERN_PROTO_TAU="${CNN_PATTERN_PROTO_TAU:-1.0}"
CNN_PATTERN_PROTO_LOSS_WEIGHT="${CNN_PATTERN_PROTO_LOSS_WEIGHT:-0.1}"
RUN_NAME="${RUN_NAME:-pamm_w${PAMM_WIN_SIZE}_p${PAMM_PATCH_SIZE}_s${PAMM_PATCH_STRIDE}_revin${PAMM_USE_REVIN}_BATCH_SIZE${PAMM_BATCH_SIZE}_SPLITS${SPLITS}_$(date +%Y%m%d_%H%M%S)}"

OUTPUT_DIR="${REPO_ROOT}/model_pamm/results_tsb_ad_runs/${RUN_NAME}"
SCORE_DIR="${OUTPUT_DIR}/scores"
METRICS_DIR="${OUTPUT_DIR}/metrics"
SERVER_LOG_DIR="${OUTPUT_DIR}/server_logs"
SERVER_LOG_FILE="${SERVER_LOG_DIR}/${RUN_NAME}.log"

mkdir -p "${SERVER_LOG_DIR}"

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"

echo "[$(date '+%F %T')] Start PAMM benchmark-phase run"
echo "[$(date '+%F %T')] RUN_NAME=${RUN_NAME}"
echo "[$(date '+%F %T')] SPLITS=${SPLITS}"
echo "[$(date '+%F %T')] PHASE=${PHASE}"
echo "[$(date '+%F %T')] DATASETS=${DATASETS}"
echo "[$(date '+%F %T')] MODELS=${MODELS}"
echo "[$(date '+%F %T')] GPU_ID=${GPU_ID}"
echo "[$(date '+%F %T')] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[$(date '+%F %T')] PAMM_WIN_SIZE=${PAMM_WIN_SIZE}"
echo "[$(date '+%F %T')] PAMM_LR=${PAMM_LR}"
echo "[$(date '+%F %T')] PAMM_BATCH_SIZE=${PAMM_BATCH_SIZE}"
echo "[$(date '+%F %T')] PAMM_EPOCHS=${PAMM_EPOCHS}"
echo "[$(date '+%F %T')] PAMM_D_MODEL=${PAMM_D_MODEL}"
echo "[$(date '+%F %T')] PAMM_PATCH_SIZE=${PAMM_PATCH_SIZE}"
echo "[$(date '+%F %T')] PAMM_PATCH_STRIDE=${PAMM_PATCH_STRIDE}"
echo "[$(date '+%F %T')] PAMM_USE_LOCALITY_BIAS=${PAMM_USE_LOCALITY_BIAS}"
echo "[$(date '+%F %T')] PAMM_CONTRAST_WEIGHT=${PAMM_CONTRAST_WEIGHT}"
echo "[$(date '+%F %T')] PAMM_USE_REVIN=${PAMM_USE_REVIN}"
echo "[$(date '+%F %T')] POINT_AGGREGATE_MODE=${POINT_AGGREGATE_MODE}"
echo "[$(date '+%F %T')] POINT_CENTER_POWER=${POINT_CENTER_POWER}"
echo "[$(date '+%F %T')] POINT_GAUSSIAN_SIGMA=${POINT_GAUSSIAN_SIGMA}"
echo "[$(date '+%F %T')] SCORE_PROJECTION_MODE=${SCORE_PROJECTION_MODE}"
echo "[$(date '+%F %T')] SCORE_PROJECTION_CENTER_POWER=${SCORE_PROJECTION_CENTER_POWER}"
echo "[$(date '+%F %T')] DIFF_PREJUDGE_ENABLED=${DIFF_PREJUDGE_ENABLED}"
echo "[$(date '+%F %T')] DIFF_PREJUDGE_QUANTILE=${DIFF_PREJUDGE_QUANTILE}"
echo "[$(date '+%F %T')] DIFF_PREJUDGE_COSINE_QUANTILE=${DIFF_PREJUDGE_COSINE_QUANTILE}"
echo "[$(date '+%F %T')] DIFF_PREJUDGE_MARGIN=${DIFF_PREJUDGE_MARGIN}"
echo "[$(date '+%F %T')] DIFF_PREJUDGE_SUPPRESSION=${DIFF_PREJUDGE_SUPPRESSION}"
echo "[$(date '+%F %T')] PATCH_CHANNEL_TOPK_WEIGHT=${PATCH_CHANNEL_TOPK_WEIGHT}"
echo "[$(date '+%F %T')] PATCH_CHANNEL_TOPK_RATIO=${PATCH_CHANNEL_TOPK_RATIO}"
echo "[$(date '+%F %T')] CNN_PATTERN_ENABLED=${CNN_PATTERN_ENABLED}"
echo "[$(date '+%F %T')] CNN_PATTERN_SCORE_WEIGHT=${CNN_PATTERN_SCORE_WEIGHT}"
echo "[$(date '+%F %T')] CNN_PATTERN_HIDDEN_DIM=${CNN_PATTERN_HIDDEN_DIM}"
echo "[$(date '+%F %T')] CNN_PATTERN_EMBEDDING_DIM=${CNN_PATTERN_EMBEDDING_DIM}"
echo "[$(date '+%F %T')] CNN_PATTERN_NUM_CONTEXTS=${CNN_PATTERN_NUM_CONTEXTS}"
echo "[$(date '+%F %T')] CNN_PATTERN_PROTO_TAU=${CNN_PATTERN_PROTO_TAU}"
echo "[$(date '+%F %T')] CNN_PATTERN_PROTO_LOSS_WEIGHT=${CNN_PATTERN_PROTO_LOSS_WEIGHT}"
echo "[$(date '+%F %T')] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[$(date '+%F %T')] SERVER_LOG=${SERVER_LOG_FILE}"
echo "[$(date '+%F %T')] RUNNER_SCRIPT=${RUNNER_SCRIPT}"

read -r -a SPLIT_ARGS <<< "${SPLITS}"
read -r -a DATASET_ARGS <<< "${DATASETS}"
read -r -a MODEL_ARGS <<< "${MODELS}"

if [[ "${PHASE}" == "both" ]]; then
  PHASE_ARGS=(tuning eval)
else
  PHASE_ARGS=("${PHASE}")
fi

RUNNER_ARGS=()
runner_supports_arg() {
  local arg_name="$1"
  grep -q -- "${arg_name}" "${RUNNER_SCRIPT}"
}

add_runner_arg() {
  local arg_name="$1"
  local arg_value="${2:-}"
  if runner_supports_arg "${arg_name}"; then
    if [[ -n "${arg_value}" ]]; then
      RUNNER_ARGS+=("${arg_name}" "${arg_value}")
    else
      RUNNER_ARGS+=("${arg_name}")
    fi
  else
    echo "[$(date '+%F %T')] WARNING: ${RUNNER_SCRIPT} does not support ${arg_name}; skipping it." >&2
  fi
}

if [[ "${OVERWRITE}" == "1" ]]; then
  add_runner_arg --overwrite
fi
if [[ "${NO_SAVE_SCORES}" == "1" ]]; then
  add_runner_arg --no_save_scores
fi
if [[ -n "${PAMM_WIN_SIZE}" ]]; then
  add_runner_arg --pamm_win_size "${PAMM_WIN_SIZE}"
fi
if [[ -n "${PAMM_LR}" ]]; then
  add_runner_arg --pamm_lr "${PAMM_LR}"
fi
if [[ -n "${PAMM_BATCH_SIZE}" ]]; then
  add_runner_arg --pamm_batch_size "${PAMM_BATCH_SIZE}"
fi
if [[ -n "${PAMM_EPOCHS}" ]]; then
  add_runner_arg --pamm_epochs "${PAMM_EPOCHS}"
fi
if [[ -n "${PAMM_D_MODEL}" ]]; then
  add_runner_arg --pamm_d_model "${PAMM_D_MODEL}"
fi
if [[ -n "${PAMM_PATCH_SIZE}" ]]; then
  add_runner_arg --pamm_patch_size "${PAMM_PATCH_SIZE}"
fi
if [[ -n "${PAMM_PATCH_STRIDE}" ]]; then
  add_runner_arg --pamm_patch_stride "${PAMM_PATCH_STRIDE}"
fi
if [[ -n "${PAMM_USE_LOCALITY_BIAS}" ]]; then
  add_runner_arg --pamm_use_locality_bias "${PAMM_USE_LOCALITY_BIAS}"
fi
if [[ -n "${PAMM_CONTRAST_WEIGHT}" ]]; then
  add_runner_arg --pamm_contrast_weight "${PAMM_CONTRAST_WEIGHT}"
fi
if [[ -n "${PAMM_USE_REVIN}" ]]; then
  add_runner_arg --pamm_use_revin "${PAMM_USE_REVIN}"
fi
if [[ -n "${POINT_AGGREGATE_MODE}" ]]; then
  add_runner_arg --pamm_point_aggregate_mode "${POINT_AGGREGATE_MODE}"
fi
if [[ -n "${POINT_CENTER_POWER}" ]]; then
  add_runner_arg --pamm_point_center_power "${POINT_CENTER_POWER}"
fi
if [[ -n "${POINT_GAUSSIAN_SIGMA}" ]]; then
  add_runner_arg --pamm_point_gaussian_sigma "${POINT_GAUSSIAN_SIGMA}"
fi
if [[ -n "${SCORE_PROJECTION_MODE}" ]]; then
  add_runner_arg --pamm_score_projection_mode "${SCORE_PROJECTION_MODE}"
fi
if [[ -n "${SCORE_PROJECTION_CENTER_POWER}" ]]; then
  add_runner_arg --pamm_score_projection_center_power "${SCORE_PROJECTION_CENTER_POWER}"
fi
if [[ -n "${DIFF_PREJUDGE_ENABLED}" ]]; then
  add_runner_arg --pamm_diff_prejudge_enabled "${DIFF_PREJUDGE_ENABLED}"
fi
if [[ -n "${DIFF_PREJUDGE_QUANTILE}" ]]; then
  add_runner_arg --pamm_diff_prejudge_quantile "${DIFF_PREJUDGE_QUANTILE}"
fi
if [[ -n "${DIFF_PREJUDGE_COSINE_QUANTILE}" ]]; then
  add_runner_arg --pamm_diff_prejudge_cosine_quantile "${DIFF_PREJUDGE_COSINE_QUANTILE}"
fi
if [[ -n "${DIFF_PREJUDGE_MARGIN}" ]]; then
  add_runner_arg --pamm_diff_prejudge_margin "${DIFF_PREJUDGE_MARGIN}"
fi
if [[ -n "${DIFF_PREJUDGE_SUPPRESSION}" ]]; then
  add_runner_arg --pamm_diff_prejudge_suppression "${DIFF_PREJUDGE_SUPPRESSION}"
fi
if [[ -n "${PATCH_CHANNEL_TOPK_WEIGHT}" ]]; then
  add_runner_arg --pamm_patch_channel_topk_weight "${PATCH_CHANNEL_TOPK_WEIGHT}"
fi
if [[ -n "${PATCH_CHANNEL_TOPK_RATIO}" ]]; then
  add_runner_arg --pamm_patch_channel_topk_ratio "${PATCH_CHANNEL_TOPK_RATIO}"
fi
if [[ -n "${CNN_PATTERN_ENABLED}" ]]; then
  add_runner_arg --pamm_cnn_pattern_enabled "${CNN_PATTERN_ENABLED}"
fi
if [[ -n "${CNN_PATTERN_SCORE_WEIGHT}" ]]; then
  add_runner_arg --pamm_cnn_pattern_score_weight "${CNN_PATTERN_SCORE_WEIGHT}"
fi
if [[ -n "${CNN_PATTERN_HIDDEN_DIM}" ]]; then
  add_runner_arg --pamm_cnn_pattern_hidden_dim "${CNN_PATTERN_HIDDEN_DIM}"
fi
if [[ -n "${CNN_PATTERN_EMBEDDING_DIM}" ]]; then
  add_runner_arg --pamm_cnn_pattern_embedding_dim "${CNN_PATTERN_EMBEDDING_DIM}"
fi
if [[ -n "${CNN_PATTERN_NUM_CONTEXTS}" ]]; then
  add_runner_arg --pamm_cnn_pattern_num_contexts "${CNN_PATTERN_NUM_CONTEXTS}"
fi
if [[ -n "${CNN_PATTERN_PROTO_TAU}" ]]; then
  add_runner_arg --pamm_cnn_pattern_proto_tau "${CNN_PATTERN_PROTO_TAU}"
fi
if [[ -n "${CNN_PATTERN_PROTO_LOSS_WEIGHT}" ]]; then
  add_runner_arg --pamm_cnn_pattern_proto_loss_weight "${CNN_PATTERN_PROTO_LOSS_WEIGHT}"
fi

run_one_phase() {
  local split="$1"
  local phase="$2"
  shift 2

  echo "[$(date '+%F %T')] Running split=${split}, phase=${phase}"
  python "${RUNNER_SCRIPT}" \
    --split "${split}" \
    --phase "${phase}" \
    --datasets "${DATASET_ARGS[@]}" \
    --models "${MODEL_ARGS[@]}" \
    --score_dir "${SCORE_DIR}" \
    --metrics_dir "${METRICS_DIR}" \
    "${RUNNER_ARGS[@]}" \
    "$@"
}

{
  for split in "${SPLIT_ARGS[@]}"; do
    for phase in "${PHASE_ARGS[@]}"; do
      case "${phase}" in
        tuning|eval)
          run_one_phase "${split}" "${phase}" "$@"
          ;;
        *)
          echo "Unsupported PHASE=${phase}. Use tuning, eval, or both." >&2
          exit 2
          ;;
      esac
    done
  done
} 2>&1 | tee "${SERVER_LOG_FILE}"

echo "[$(date '+%F %T')] Finished."
echo "[$(date '+%F %T')] Metrics root: ${METRICS_DIR}"
echo "[$(date '+%F %T')] Scores root: ${SCORE_DIR}"
