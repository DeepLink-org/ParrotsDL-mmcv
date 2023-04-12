#include "parrots_call_mluops.hpp"

#include <parrots/foundation/cnnlhandle.hpp>


// DataType
inline mluOpDataType_t getMluOpDataType(parrots::ValueType vt) {
  switch (vt.prim()) {
    case Prim::Float16:
      return MLUOP_DTYPE_HALF;
    case Prim::Float32:
      return MLUOP_DTYPE_FLOAT;
    case Prim::Int16:
      return MLUOP_DTYPE_INT16;
    case Prim::Int32:
      return MLUOP_DTYPE_INT32;
    case Prim::Int64:
      return MLUOP_DTYPE_INT64;
    case Prim::Int8:
      return MLUOP_DTYPE_INT8;
    case Prim::Uint8:
      return MLUOP_DTYPE_INT8;
    case Prim::Bool:
      return MLUOP_DTYPE_BOOL;
    default:
      PARROTS_NOTSUPPORTED << "Unsupported data type for MLUOPS: " << vt.name();
  }
}

// Laytout
mluOpTensorLayout_t getMluOpSuggestLayout(const DArrayLite& input) {
  auto suggest_memory_format = input.spec().probableMemoryFormat();
  mluOpTensorLayout_t layout = MLUOP_LAYOUT_ARRAY;
  switch (input.ndims()) {
    case 4:
      layout = (suggest_memory_format == MemoryFormat::ChannelsLast)
                   ? MLUOP_LAYOUT_NHWC
                   : MLUOP_LAYOUT_NCHW;
      break;
    default:
      layout = MLUOP_LAYOUT_ARRAY;
  }
  return layout;
}

// Descriptors
void MluOpTensorDescriptor::set(DArrayLite t) {
  mluOpDataType_t data_type = getMluOpDataType(t.elemType());
  mluOpTensorLayout_t layout = getMluOpSuggestLayout(t);
  int t_dim = t.ndims();
  std::vector<int> dim_array;
  if (t_dim == 0) {
    dim_array.push_back(
        1);  // ScalarTensor(0-dim 1-item Tensor) view like size = 1 as default;
  } else {
    for (int i = 0; i < t_dim; i++) {
      dim_array.push_back(static_cast<int>(t.shape().dim(i)));
    }
  }
  set_desc(t, layout, data_type, dim_array);
}

void MluOpTensorDescriptor::set_desc(const DArrayLite& t,
                                     mluOpTensorLayout_t layout,
                                     mluOpDataType_t dtype,
                                     std::vector<int>& dims) {
  int dimNb = dims.size();
  mluOpSetTensorDescriptor(desc_, layout, dtype, dimNb, dims.data());
}

// Handles
std::once_flag mmcv_mluop_init_flag;
std::mutex mmcv_mluop_mutex;
static std::vector<MluOpHandle> mmcv_mluop_handles;

mluOpHandle_t mluOpGetCurrentHandle(CambContext& ctx) {
  std::call_once(mmcv_mluop_init_flag,
                 [&]()  // Init mmcv_mluop_handles 1-device <-> 1-handle
                 {
                   CambDevice device = dynamic_cast<const DeviceProxyT<CambDevice>&>(ctx.getProxy()).device();
                   int num_device = device.deviceCount();
                   mmcv_mluop_handles.resize(num_device);
                 });

  int device_index = -1;
  device_index = ctx.getProxy().deviceId();

  std::lock_guard<std::mutex> mmcv_mluop_guard(mmcv_mluop_mutex);
  auto queue = ctx.getStream().native();
  mmcv_mluop_handles[device_index].setQueue(queue);
  return mmcv_mluop_handles[device_index].handle;
}

