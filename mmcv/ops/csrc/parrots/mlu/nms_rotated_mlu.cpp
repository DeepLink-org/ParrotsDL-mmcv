#include <parrots/darray/darraymath.hpp>
#include <parrots_mlu_helper.hpp>
#include <parrots_call_mluops.hpp>


#include "mlu_op.h"

using namespace parrots;

DArrayLite nms_rotated_mlu(CambContext& ctx, const DArrayLite boxes, const DArrayLite scores,
                           const float iou_threshold) {
  if (boxes.size() == 0) {
    DArrayLite output = ctx.createDArrayLite(DArraySpec::array(Prim::Int64, DArrayShape(0)));
    return output;
  }

  int boxes_num = boxes.dim(0);
  auto boxes_ = boxes;
  auto scores_ = scores;
  DArrayLite output = ctx.createDArrayLite(DArraySpec::array(Prim::Int32, DArrayShape(boxes_num)));
  DArrayLite output_size = ctx.createDArrayLite(DArraySpec::array(Prim::Int32, DArrayShape(1)));

  MluOpTensorDescriptor boxes_desc, scores_desc, output_desc;
  boxes_desc.set(boxes_);
  scores_desc.set(scores_);
  output_desc.set(output);

  size_t workspace_size(0);
  auto handle = mluOpGetCurrentHandle(ctx);
  mluOpGetNmsRotatedWorkspaceSize(handle, boxes_desc.desc(), &workspace_size);
  DArrayLite workspace = ctx.createDArrayLite(DArraySpec::array(Prim::Uint8, DArrayShape(workspace_size)));

  auto boxes_ptr = boxes_.data();
  auto scores_ptr = scores_.data();
  auto workspace_ptr = workspace.data();
  auto output_ptr = output.data();
  auto output_size_ptr = output_size.data();

  mluOpNmsRotated(handle, iou_threshold, boxes_desc.desc(), boxes_ptr,
                  scores_desc.desc(), scores_ptr, workspace_ptr, workspace_size,
                  output_desc.desc(), output_ptr, (int *)output_size_ptr);
  DArrayLite outputHost = ctx.createDArrayLite(DArraySpec::array(Prim::Int32, DArrayShape(1)), getHostProxy());
  copy(ctx, outputHost, output_size);
  ctx.getStream().synchronize();
  int output_num = *static_cast<int *>(outputHost.data());
  
  DArrayLite ret = ctx.createDArrayLite(DArraySpec::array(Prim::Int64, DArrayShape(boxes_num)));
  cast(ctx, output, ret);
  DArrayLite reout = op::slice(ctx, ret, 0, 0, output_num, 1);
  return reout;
}


void nms_rotated_parrots_mlu(CambContext& ctx, const SSElement& attr,
                             const OperatorBase::in_list_t& ins,
                             OperatorBase::out_list_t& outs) {
  float iou_threshold;
  int multi_label;
  SSAttrs(attr)
      .get("iou_threshold", iou_threshold)
      .get("multi_label", multi_label)
      .done();
  auto dets = ins[0];
  auto scores = ins[1];
  auto order = ins[2];
  auto dets_sorted = ins[3];
  auto out = nms_rotated_mlu(ctx, dets, scores, iou_threshold);
  outs.at(0) = out;
}

PARROTS_EXTENSION_REGISTER(nms_rotated)
    .attr("multi_label")
    .attr("iou_threshold")
    .input(4)
    .output(1)
    .apply(nms_rotated_parrots_mlu)
    .done();
