#include <parrots/darray/darraymath.hpp>
#include <parrots/foundation/logging.hpp>
#include <parrots_mlu_helper.hpp>

#include "parrots_nms.h"

using namespace parrots;

void KernelNms(cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
               const cnrtDataType_t data_type_input, const void *boxes_ptr,
               const void *scores_ptr, const int input_num_boxes,
               const int max_output_boxes, const float iou_threshold,
               const float offset, void *workspace_ptr, void *output_size_ptr,
               void *output_ptr);

inline int32_t getJobLimitCapability() {
  CNcontext drv_ctx;
  PARROTS_CHECKARGS(CN_SUCCESS == cnCtxGetCurrent(&drv_ctx))
      << "cnCtxGetCurrent fails";
  CNctxConfigParam ctx_conf_param;
  PARROTS_CHECKARGS(CN_SUCCESS == cnGetCtxConfigParam(drv_ctx,
                                                      CN_CTX_CONFIG_UNION_LIMIT,
                                                      &ctx_conf_param))
      << "cnGetCtxConfigParam fails.";
  return (int32_t)ctx_conf_param.unionLimit;
}

int selectUnionType(uint32_t use_job, int box_num_per_core) {
  // the box_num_per_core should be at least 256, otherwise the real IO
  // bandwidth would be very low
  while (box_num_per_core < 256 && use_job >= 4) {
    box_num_per_core *= 2;
    use_job /= 2;
  }
  return use_job;
}

static cnnlStatus_t policyFunc(cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type,
                               const int input_box_num) {
  uint32_t core_dim = getDeviceAttr(cnrtAttrMcorePerCluster);
  uint32_t job_limit = getJobLimitCapability();
  uint32_t core_number = job_limit;

  int box_num_per_core = (input_box_num + core_number - 1) / core_number;
  int use_job = selectUnionType(job_limit, box_num_per_core);
  // initiate k_type as Union1
  k_dim->x = core_dim;
  k_dim->y = 1;
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
  switch (job_limit) {
    case CN_KERNEL_CLASS_BLOCK:
    case CN_KERNEL_CLASS_UNION:
    case CN_KERNEL_CLASS_UNION2:
    case CN_KERNEL_CLASS_UNION4:
    case CN_KERNEL_CLASS_UNION8:
    case CN_KERNEL_CLASS_UNION16: {
      if (use_job < 4) {
        k_dim->x = 1;
        *k_type = CNRT_FUNC_TYPE_BLOCK;
      } else if (use_job == 4) {
        k_dim->x = core_dim;
        *k_type = CNRT_FUNC_TYPE_UNION1;
      } else {
        k_dim->x = use_job;
        *k_type = (cnrtFunctionType_t)use_job;
      }
    }; break;
    default:
      PARROTS_LOG_WARN(
          "[cnnlNms_v2]: got unsupported job limit number. "
          "Use default CN_KERNEL_CLASS_UNION1 with UNION1 task.");
  }
  return CNNL_STATUS_SUCCESS;
}

void NMSMLUKernelLauncher(CambContext &ctx, const DArrayLite &boxes,
                          const DArrayLite &scores, DArrayLite &output,
                          const float iou_threshold, const int offset) {
  // dimension parameters check
  PARROTS_CHECKARGS(boxes.ndims() == 2)
      << "boxes should be a 2d tensor, got " << boxes.ndims() << "D";
  PARROTS_CHECKARGS(boxes.dim(1) == 4)
      << "boxes should have 4 elements in dimension 1, got " << boxes.dim(1);
  PARROTS_CHECKARGS(scores.ndims() == 1)
      << "scores should be a 1d tensor, got " << scores.ndims() << "D";
  // data type check
  PARROTS_CHECKARGS(boxes.elemType() == scores.elemType())
      << "boxes should have the same type as scores";
  PARROTS_CHECKARGS(boxes.elemType() == Prim::Float32 ||
                    boxes.elemType() == Prim::Float16)
      << "data type of boxes should be Float or Half, got " << boxes.elemType();

  if (boxes.size() == 0) {
    output = ctx.createDArrayLite(boxes.spec().withElemType(Prim::Int64));
    return;
  }
  int input_num_boxes = boxes.dim(0);
  int max_output_boxes = boxes.dim(0);

  cnrtDataType_t data_type_input = getCnrtDataType(boxes.elemType());
  cnrtDim3_t k_dim;
  cnrtJobType_t k_type;
  policyFunc(&k_dim, &k_type, input_num_boxes);

  // transpose boxes (n, 4) to (4, n) for better performance
  DArrayLite boxes_tmp = t(ctx, boxes);
  DArrayLite scores_tmp = scores;
  if (!boxes_tmp.isContiguous()) {
    boxes_tmp = ctx.cloneDArrayLite(boxes_tmp);
  }
  if (!scores_tmp.isContiguous()) {
    scores_tmp = ctx.cloneDArrayLite(scores_tmp);
  }

  DArrayLite output_tmp =
      ctx.createDArrayLite(boxes_tmp.spec()
                               .withElemType(Prim::Int64)
                               .withShape(DArrayShape(max_output_boxes)));
  DArrayLite output_size = ctx.createDArrayLite(
      scores_tmp.spec().withElemType(Prim::Int32).withShape(DArrayShape(1)));

  // workspace
  size_t space_size = 0;
  const int info_num = 5;  // x1, x2, y1, y2 and score
  if (boxes_tmp.elemType() == Prim::Float16) {
    space_size = input_num_boxes * sizeof(int16_t) * info_num + sizeof(float);
  } else {
    space_size = input_num_boxes * sizeof(float) * info_num + sizeof(float);
  }
  auto workspace = ctx.createDArrayLite(DArraySpec::bytes(space_size));

  // get compute queue
  auto queue = getStreamNative<CambDevice>(ctx.getStream());
  KernelNms(k_dim, k_type, queue, data_type_input, boxes_tmp.data(),
            scores_tmp.data(), input_num_boxes, max_output_boxes, iou_threshold,
            offset, workspace.data(), output_size.data(), output_tmp.data());

  int output_num = 0;
  PARROTS_CALLCNRT(cnrtMemcpyAsync(&output_num, output_size.data(), sizeof(int),
                                   queue, cnrtMemcpyDevToHost));
  // since we need to copy then, synchronize here
  ctx.getStream().synchronize();
  DArrayLite output32 =
      ctx.createDArrayLite(boxes_tmp.spec()
                               .withElemType(Prim::Int32)
                               .withShape(DArrayShape(output_num)));
  output = ctx.createDArrayLite(boxes_tmp.spec()
                                    .withElemType(Prim::Int64)
                                    .withShape(DArrayShape(output_num)));
  PARROTS_CALLCNRT(cnrtMemcpyAsync(output32.data(), output_tmp.data(),
                                   output32.nbytes(), queue,
                                   cnrtMemcpyDevToDev));
  cast(ctx, output32, output);
}

void nms_parrots_device(CambContext &ctx, const SSElement &attr,
                        const OperatorBase::in_list_t &ins,
                        OperatorBase::out_list_t &outs) {
  float iou_threshold;
  int offset;
  SSAttrs(attr)
      .get("iou_threshold", iou_threshold)
      .get("offset", offset)
      .done();
  const auto &boxes = ins[0];
  const auto &scores = ins[1];
  auto &out = outs[0];
  NMSMLUKernelLauncher(ctx, boxes, scores, out, iou_threshold, offset);
}

REGISTER_DEVICE_IMPL(nms_impl, MLU, Arch::CAMB, nms_parrots_device);
