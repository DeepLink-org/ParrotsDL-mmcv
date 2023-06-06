
#include <parrots/compute/aten.hpp>
#include <parrots/darray/darraymath.hpp>
#include <parrots/foundation/darrayutil.hpp>
#include <parrots_mlu_helper.hpp>
using namespace parrots;


#define MIN(a, b) (((a) < (b)) ? (a) : (b))

typedef enum {
  MS_DEFORM_ATTN_FORWARD_INVALID = 0, /*!< Index is invalid. */
  MS_DEFORM_ATTN_FORWARD_DEFAULT =
      1, /*!< MLUKernelMsDeformAttnForwardDefault */
  MS_DEFORM_ATTN_FORWARD_SMALL_CHANNEL =
      2, /*!< MLUKernelMsDeformAttnForwardSmallChannel */
} MsDeformAttnForwardPolicy;

void KernelMsDeformAttnForwardDefault(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const cnrtDataType_t d_type, const char* data_value_gdram,
    const char* data_spatial_shapes_gdram,
    const char* data_level_start_index_gdram,
    const char* data_sampling_loc_gdram, const char* data_attn_weight_gdram,
    const int32_t batch_size, const int32_t num_keys, const int32_t num_heads,
    const int32_t channels, const int32_t num_levels, const int32_t num_queries,
    const int32_t num_points, char* data_col_gdram);
void KernelMsDeformAttnForwardSmallChannel(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const cnrtDataType_t d_type, const char* data_value_gdram,
    const char* data_spatial_shapes_gdram,
    const char* data_level_start_index_gdram,
    const char* data_sampling_loc_gdram, const char* data_attn_weight_gdram,
    const int32_t batch_size, const int32_t num_keys, const int32_t num_heads,
    const int32_t channels, const int32_t num_levels, const int32_t num_queries,
    const int32_t num_points, char* data_col_gdram);

typedef enum {
  MS_DEFORM_ATTN_BACKWARD_DEFAULT = 0,
  MS_DEFORM_ATTN_BACKWARD_SMALL_CHANNEL = 1,
} MsDeformAttnBackwardKernelPolicy;

MsDeformAttnBackwardKernelPolicy msDeformAttnBackwardPolicyFunc(
    const int32_t channels, const int32_t num_levels, const int32_t num_points,
    const int32_t num_heads) {
  const int32_t nram_size = getDeviceAttr(cnrtAttrNramSizePerMcore);
  const int num_hlp = num_heads * num_levels * num_points;
  int num_per_time_theory = (nram_size - num_levels * sizeof(float) -
                             3 * num_levels * sizeof(int32_t)) /
                            sizeof(float) / (8 * PAD_UP(channels, 32) + 28) /
                            PAD_UP((num_hlp), 32);
  if (num_per_time_theory >= 1) {
    return MS_DEFORM_ATTN_BACKWARD_SMALL_CHANNEL;
  }
  return MS_DEFORM_ATTN_BACKWARD_DEFAULT;
}

void KernelMsDeformAttnBackwardDefaultKernel(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const cnrtDataType_t d_type, const float* data_value,
    const int32_t* spatial_shapes, const int32_t* data_level_start_index,
    const float* data_sampling_loc, const float* data_attn_weight,
    const float* grad_output, const int32_t batch_size, const int32_t num_keys,
    const int32_t num_heads, const int32_t channels, const int32_t num_levels,
    const int32_t num_queries, const int32_t num_points, float* grad_value,
    float* grad_sampling_loc, float* grad_attn_weight);

void KernelMsDeformAttnBackwardSmallChannelsKernel(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const cnrtDataType_t d_type, const float* data_value,
    const int32_t* spatial_shapes, const int32_t* data_level_start_index,
    const float* data_sampling_loc, const float* data_attn_weight,
    const float* grad_output, const int32_t batch, const int32_t spatial_size,
    const int32_t num_heads, const int32_t channels, const int32_t num_levels,
    const int32_t num_query, const int32_t num_points, float* grad_value,
    float* grad_sampling_loc, float* grad_attn_weight);


MsDeformAttnForwardPolicy msDeformAttnForwardPolicyFunc(
    cnrtDim3_t* k_dim, cnrtFunctionType_t* k_type, const int32_t batch_size,
    const int32_t num_keys, const int32_t num_heads, const int32_t channels,
    const int32_t num_levels, const int32_t num_queries,
    const int32_t num_points) {
  k_dim->x = getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim->y =
      MIN((batch_size * num_queries * num_heads + k_dim->x - 1) / k_dim->x,
          getDeviceAttr(cnrtAttrClusterCount));
  k_dim->z = 1;
#if __BANG_ARCH__ == 520
  *k_type = CNRT_FUNC_TYPE_BLOCK;
#else
  *k_type = CNRT_FUNC_TYPE_UNION1;
#endif

  int32_t nram_size = getDeviceAttr(cnrtAttrNramSizePerMcore);
  if (num_levels * num_points * 3 * sizeof(int32_t) > nram_size) {
    return MS_DEFORM_ATTN_FORWARD_DEFAULT;
  } else if (channels > nram_size / 12 / sizeof(float) || channels > 96 ||
             channels < 16) {
    return MS_DEFORM_ATTN_FORWARD_DEFAULT;
  } else {
    return MS_DEFORM_ATTN_FORWARD_SMALL_CHANNEL;
  }
}


static void policyFuncBackward(const int32_t batch_size,
                               const int32_t num_queries,
                               const int32_t num_heads,
                               const int32_t num_levels,
                               cnrtFunctionType_t* k_type, cnrtDim3_t* k_dim) {
  size_t cluster_limit = getDeviceAttr(cnrtAttrClusterCount);
  size_t core_limit = getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim->x = core_limit;
  int32_t total_num = batch_size * num_queries * num_heads * num_levels;
  size_t total_num_align = CEIL_ALIGN(total_num, core_limit);
  k_dim->y = (total_num_align / core_limit) > cluster_limit
                 ? cluster_limit
                 : (total_num_align / core_limit);
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
}

DArrayLite ms_deform_attn_mlu_forward(CambContext& ctx, const DArrayLite& value,
                                  const DArrayLite& spatial_shapes,
                                  const DArrayLite& level_start_index,
                                  const DArrayLite& sampling_loc,
                                  const DArrayLite& attn_weight,
                                  const int im2col_step) {

  // check datatype
  PARROTS_CHECKARGS(value.elemType() == Prim::Float32)
              <<"value type should be Float, got "<< value.elemType()<< ".";
  PARROTS_CHECKARGS(spatial_shapes.elemType() == Prim::Int32 ||
               spatial_shapes.elemType() == Prim::Int64)
               <<"spatial_shapes type should be Int, got "
               <<spatial_shapes.elemType()<< ".";
  PARROTS_CHECKARGS(level_start_index.elemType() == Prim::Int32 ||
               level_start_index.elemType() == Prim::Int64)
               <<"level_start_index type should be Int, got "
               <<level_start_index.elemType()<< ".";
  PARROTS_CHECKARGS(sampling_loc.elemType() == Prim::Float32)
               <<"sampling_loc type should be Float, got "
               <<sampling_loc.elemType()<< ".";
  PARROTS_CHECKARGS(attn_weight.elemType() == Prim::Float32)
              <<"attn_weight type should be Float, got "
              <<attn_weight.elemType()<< ".";

  // check shape
  PARROTS_CHECKARGS(value.ndims() == 4)<< "value should be a 4d tensor, got "
              <<value.ndims()<< "D.";
  PARROTS_CHECKARGS(spatial_shapes.ndims() == 2)
              <<"spatial_shapes should be a 2d tensor, got "
              <<spatial_shapes.ndims()<< "D.";
  PARROTS_CHECKARGS(level_start_index.ndims() == 1)
              <<"level_start_index should be a 1d tensor, got "
              <<level_start_index.ndims()<< "D.";
  PARROTS_CHECKARGS(sampling_loc.ndims() == 6)
              <<"sampling_loc should be a 6d tensor, got "
              << sampling_loc.ndims()<<"D.";
  PARROTS_CHECKARGS(attn_weight.ndims() == 5)
              << "attn_weight should be a 5d tensor, got "
              <<attn_weight.ndims()<< "D.";

  const int batch_size = value.dim(0);
  const int num_keys = value.dim(1);
  const int num_heads = value.dim(2);
  const int channels = value.dim(3);
  const int num_levels = spatial_shapes.dim(0);
  const int num_queries = sampling_loc.dim(1);
  const int num_points = sampling_loc.dim(4);

  PARROTS_CHECKARGS(spatial_shapes.dim(1) == 2)
              <<"the 2nd dimensions of spatial_shapes should be 2, got "
              <<spatial_shapes.dim(1)<< ".";
  PARROTS_CHECKARGS(sampling_loc.dim(5) == 2)
              <<"the 6th dimensions of sampling_loc should be 2, got "
              <<sampling_loc.dim(5)<< ".";
  PARROTS_CHECKARGS(sampling_loc.dim(0) == batch_size)
              <<"the 1st dimensions of sampling_loc should be batch_size, "
              <<"but now the 1st dimension of sampling_loc is "
              <<sampling_loc.dim(0)<< ", and batch_size is "<< batch_size<< ".";
  PARROTS_CHECKARGS(attn_weight.dim(0) == batch_size)
              <<"the 1st dimensions of attn_weight should be batch_size, "
              <<"but now the 1st dimension of attn_weight is "
              <<attn_weight.dim(0)<< ", and batch_size is "<< batch_size<< ".";
  PARROTS_CHECKARGS(sampling_loc.dim(2) == num_heads)
              <<"the 3rd dimensions of sampling_loc should be num_heads, "
              <<"but now the 3rd dimension of sampling_loc is "
              <<sampling_loc.dim(2)<< ", and num_heads is "<< num_heads<< ".";
  PARROTS_CHECKARGS(attn_weight.dim(2) == num_heads)
              <<"the 3rd dimensions of attn_weight should be num_heads, "
              <<"but now the 3rd dimension of attn_weight is "
              <<attn_weight.dim(2)<< ", and num_heads is "<< num_heads<< ".";
  PARROTS_CHECKARGS(level_start_index.dim(0) == num_levels)
              <<"the 1st dimensions of level_start_index should be num_levels, "
              <<"but now the 1st dimension of level_start_index is "
              <<level_start_index.dim(0)<< ", and num_levels is "<< num_levels<<".";
  PARROTS_CHECKARGS(sampling_loc.dim(3) == num_levels)
              <<"the 4th dimensions of sampling_loc should be num_levels, "
              <<"but now the 4th dimension of sampling_loc is "
              <<sampling_loc.dim(3)<< ", and num_levels is "<< num_levels<< ".";
  PARROTS_CHECKARGS(attn_weight.dim(3) == num_levels)
              <<"the 4th dimensions of attn_weight should be num_levels, "
              <<"but now the 4th dimension of attn_weight is "
              <<attn_weight.dim(3)<< ", and num_levels is "<< num_levels<< ".";
  PARROTS_CHECKARGS(attn_weight.dim(1) == num_queries)
              <<"the 2nd dimensions of attn_weight should be num_queries, "
              <<"but now the 2nd dimension of attn_weight is "
              <<attn_weight.dim(1)<< ", and num_queries is "<< num_queries<< ".";
  PARROTS_CHECKARGS(attn_weight.dim(4) == num_points)
              <<"the 5th dimensions of attn_weight should be num_points, "
              <<"but now the 5th dimension of attn_weight is "
              <<attn_weight.dim(4)<< ", and num_points is "<< num_points<< ".";

  auto output = ctx.createDArrayLite(type_<float>(),
                            DArrayShape(batch_size, num_queries, num_heads, channels));
  
  // large tensor check
  const size_t max_input_size = 2147483648;
  PARROTS_CHECKARGS(value.size() < max_input_size)
              <<"value element num should be less than 2^31, got "<< value.size()
              <<".";
  PARROTS_CHECKARGS(sampling_loc.size() < max_input_size)
              <<"sampling_loc element num should be less than 2^31, got "
              <<sampling_loc.size()<< ".";
  PARROTS_CHECKARGS(output.size() < max_input_size)
              <<"output element num should be less than 2^31, got "
              <<output.size()<< ".";

  // check zero element
  PARROTS_CHECKARGS(batch_size != 0)<< "batch_size should not be zero";
  PARROTS_CHECKARGS(num_heads != 0)<< "num_heads should not be zero";
  PARROTS_CHECKARGS(channels != 0)<< "channels should not be zero";
  PARROTS_CHECKARGS(num_queries != 0)<< "num_queries should not be zero";

  //normal(ctx, 0.5, 0.5, output);
  //std::cout<<"!!! MLUKernelMsDeformAttnForwardDefault return a fake tensor"<<std::endl;
  //return output;
  
  if (num_keys == 0 || num_levels == 0 || num_points == 0) {
    return output;
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  MsDeformAttnForwardPolicy policy = msDeformAttnForwardPolicyFunc(
      &k_dim, &k_type, batch_size, num_keys, num_heads, channels, num_levels,
      num_queries, num_points);


  auto queue = ctx.getStream().native();
  auto spatial_shapes_ = spatial_shapes;   //.to(Prim::Int32);
  auto level_start_index_ = level_start_index; //.to(Prim::Int32);



  auto value_ptr = value.data();

  auto spatial_shapes_ptr = spatial_shapes.data();
  auto level_start_index_ptr = level_start_index_.data();
  auto sampling_loc_ptr = sampling_loc.data();
  auto attn_weight_ptr = attn_weight.data();
  auto output_ptr = output.data();;

  cnrtDataType_t data_type = getCnrtDataType(value.elemType());


  switch (policy) {
    default: {
      //std::cout << "MsDeformAttnForward Policy not supported";
    }; break;
    case MS_DEFORM_ATTN_FORWARD_DEFAULT: {
      //std::cout << "Launch Kernel MLUKernelMsDeformAttnForwardDefault<<<"
      //            << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";
      KernelMsDeformAttnForwardDefault(
          k_dim, k_type, queue, data_type, (char*)value_ptr,
          (char*)spatial_shapes_ptr, (char*)level_start_index_ptr,
          (char*)sampling_loc_ptr, (char*)attn_weight_ptr, batch_size, num_keys,
          num_heads, channels, num_levels, num_queries, num_points,
          (char*)output_ptr);
      break;
    }
    case MS_DEFORM_ATTN_FORWARD_SMALL_CHANNEL: {
      //std::cout << "Launch Kernel MLUKernelMsDeformAttnForwardSmallChannel<<<"
      //            << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";
      KernelMsDeformAttnForwardSmallChannel(
          k_dim, k_type, queue, data_type, (char*)value_ptr,
          (char*)spatial_shapes_ptr, (char*)level_start_index_ptr,
          (char*)sampling_loc_ptr, (char*)attn_weight_ptr, batch_size, num_keys,
          num_heads, channels, num_levels, num_queries, num_points,
          (char*)output_ptr);
      break;
    }
  }

  output = output.view({batch_size, num_queries, num_heads * channels});
  return output;
}

void ms_deform_attn_mlu_backward(CambContext& ctx, 
    const DArrayLite& value, const DArrayLite& spatial_shapes,
    const DArrayLite& level_start_index, const DArrayLite& sampling_loc,
    const DArrayLite& attn_weight, const DArrayLite& grad_output, DArrayLite& grad_value,
    DArrayLite& grad_sampling_loc, DArrayLite& grad_attn_weight,
    const int im2col_step) {

  // check datatype
  PARROTS_CHECKARGS(value.elemType() == Prim::Float32)
              <<"value type should be Float, got "<< value.elemType()<< ".";
  PARROTS_CHECKARGS(spatial_shapes.elemType() == Prim::Int32 ||
               spatial_shapes.elemType() == Prim::Int64)
              <<"spatial_shapes type should be Int, got "
              <<spatial_shapes.elemType()<< ".";
  PARROTS_CHECKARGS(level_start_index.elemType() == Prim::Int32 ||
               level_start_index.elemType() == Prim::Int64)
              <<"level_start_index type should be Int, got "
              <<level_start_index.elemType()<< ".";
  PARROTS_CHECKARGS(sampling_loc.elemType() == Prim::Float32)
              <<"sampling_loc type should be Float, got "
              <<sampling_loc.elemType()<< ".";
  PARROTS_CHECKARGS(attn_weight.elemType() == Prim::Float32)
              <<"attn_weight type should be Float, got "
              <<attn_weight.elemType()<< ".";
  PARROTS_CHECKARGS(grad_output.elemType() == Prim::Float32)
              <<"grad_output type should be Float, got "
              <<grad_output.elemType()<< ".";

  const int batch_size = value.dim(0);
  const int num_keys = value.dim(1);
  const int num_heads = value.dim(2);
  const int channels = value.dim(3);
  const int num_levels = spatial_shapes.dim(0);
  const int num_queries = sampling_loc.dim(1);
  const int num_points = sampling_loc.dim(4);
  
  // Check shape.
  PARROTS_CHECKARGS(spatial_shapes.dim(1) == 2)
              <<"the 2nd dimensions of spatial_shapes should be 2, got "
              <<spatial_shapes.dim(1)<< ".";

  PARROTS_CHECKARGS(level_start_index.dim(0) == num_levels)
              <<"the 1st dimensions of level_start_index should be num_levels, "
              <<"but now the 1st dimension of level_start_index is "
              <<level_start_index.dim(0)<< ", and num_levels is "<< num_levels
              <<".";

  PARROTS_CHECKARGS(sampling_loc.dim(0) == batch_size)
              <<"the 1st dimensions of sampling_loc should be batch_size, "
              <<"but now the 1st dimension of sampling_loc is "
              <<sampling_loc.dim(0)<< ", and batch_size is "<< batch_size<< ".";
  PARROTS_CHECKARGS(sampling_loc.dim(2) == num_heads)
              <<"the 3rd dimensions of sampling_loc should be num_heads, "
              <<"but now the 3rd dimension of sampling_loc is "
              <<sampling_loc.dim(2)<< ", and num_heads is "
              << num_heads<< ".";
  PARROTS_CHECKARGS(sampling_loc.dim(3) == num_levels)
              <<"the 4th dimensions of sampling_loc should be num_levels, "
              <<"but now the 4th dimension of sampling_loc is "
              <<sampling_loc.dim(3)
              << ", and num_levels is "<< num_levels<< ".";
  PARROTS_CHECKARGS(sampling_loc.dim(5) == 2)
              <<"the 6th dimensions of sampling_loc should be 2, got "
              <<sampling_loc.dim(5)<< ".";

  PARROTS_CHECKARGS(attn_weight.dim(0) == batch_size)
              <<"the 1st dimensions of attn_weight should be batch_size, "
              <<"but now the 1st dimension of attn_weight is "
              <<attn_weight.dim(0)<< ", and batch_size is "<< batch_size<< ".";
  PARROTS_CHECKARGS(attn_weight.dim(1) == num_queries)
              <<"the 2nd dimensions of attn_weight should be num_queries, "
              <<"but now the 2nd dimension of attn_weight is "
              <<attn_weight.dim(1)
              << ", and num_queries is "<< num_queries<< ".";

  PARROTS_CHECKARGS(attn_weight.dim(2) == num_heads)
              <<"the 3rd dimensions of attn_weight should be "<< num_heads
              <<"but now the 3rd dimension of attn_weight is "
              <<attn_weight.dim(2)
              << ", and num_heads is "<< num_heads<< ".";
  PARROTS_CHECKARGS(attn_weight.dim(3) == num_levels)
              <<"the 4th dimensions of attn_weight should be num_levels, "
              <<"but now the 4th dimension of attn_weight is "
              <<attn_weight.dim(3)
              << ", and num_levels is "<< num_levels<< ".";
  PARROTS_CHECKARGS(attn_weight.dim(4) == num_points)
              <<"the 5th dimensions of attn_weight should be "<< num_points
              <<"but now the 5th dimension of attn_weight is "
              <<attn_weight.dim(4)
              << ", and num_points is "<< num_points<< ".";

  PARROTS_CHECKARGS(grad_output.dim(0) == batch_size)
              <<"the 1st dimensions of grad_output should be "<<batch_size 
              <<"but now the 1st dimension of grad_output is "
              <<grad_output.dim(0)
              << ", and batch_size is "<< batch_size<< ".";
  PARROTS_CHECKARGS(grad_output.dim(1) == num_queries)
              <<"the 2nd dimensions of grad_output should be "<<num_queries 
              <<"but now the 2nd dimension of grad_output is "
              <<grad_output.dim(1)
              << ", and num_queries is "<< num_queries<< ".";
  PARROTS_CHECKARGS(grad_output.dim(2) == num_heads * channels)
              <<"the 3rd dimensions of grad_output should be "<<num_heads * channels
              <<"but now the 3rd dimension of grad_output is "<< grad_output.dim(2)
              <<", and num_heads * channels is "<< num_heads * channels<< ".";

  // check zero element
  PARROTS_CHECKARGS(batch_size != 0)<< "The batch_size is zero.";
  PARROTS_CHECKARGS(channels != 0)<< "The channels is zero.";
  PARROTS_CHECKARGS(num_keys != 0)<< "The num_keys is zero.";
  PARROTS_CHECKARGS(num_heads != 0)<< "The num_heads is zero.";
  PARROTS_CHECKARGS(num_queries != 0)<< "The num_queries is zero.";

  if (num_levels == 0 || num_points == 0) {
    return;
  }


  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFuncBackward(batch_size, num_queries, num_heads, num_levels, &k_type,
                     &k_dim);


  auto queue = ctx.getStream().native();


  auto value_ptr = value.data();
  auto spatial_shapes_ptr = spatial_shapes.data();
  auto level_start_index_ptr = level_start_index.data();
  auto sampling_loc_ptr = sampling_loc.data();
  auto attn_weight_ptr = attn_weight.data();
  auto grad_output_ptr = grad_output.data();
  auto grad_value_ptr = grad_value.data();
  auto grad_sampling_loc_ptr = grad_sampling_loc.data();
  auto grad_attn_weight_ptr = grad_attn_weight.data();


  cnrtDataType_t data_type =  getCnrtDataType(value.elemType());


  MsDeformAttnBackwardKernelPolicy kernelPolicy =
      msDeformAttnBackwardPolicyFunc(channels, num_levels, num_points,
                                     num_heads);
  switch (kernelPolicy) {
    default: {
      std::cout << "NotImplemented.";
    } break;
    case MS_DEFORM_ATTN_BACKWARD_DEFAULT: {
      KernelMsDeformAttnBackwardDefaultKernel(
          k_dim, k_type, queue, data_type, (float*)value_ptr,
          (int32_t*)spatial_shapes_ptr, (int32_t*)level_start_index_ptr,
          (float*)sampling_loc_ptr, (float*)attn_weight_ptr,
          (float*)grad_output_ptr, batch_size, num_keys, num_heads, channels,
          num_levels, num_queries, num_points, (float*)grad_value_ptr,
          (float*)grad_sampling_loc_ptr, (float*)grad_attn_weight_ptr);
    } break;
    case MS_DEFORM_ATTN_BACKWARD_SMALL_CHANNEL: {
      KernelMsDeformAttnBackwardSmallChannelsKernel(
          k_dim, k_type, queue, data_type, (float*)value_ptr,
          (int32_t*)spatial_shapes_ptr, (int32_t*)level_start_index_ptr,
          (float*)sampling_loc_ptr, (float*)attn_weight_ptr,
          (float*)grad_output_ptr, batch_size, num_keys, num_heads, channels,
          num_levels, num_queries, num_points, (float*)grad_value_ptr,
          (float*)grad_sampling_loc_ptr, (float*)grad_attn_weight_ptr);
    } break;
  }
}

void ms_deform_attn_mlu_forward_camb_parrots(CambContext& ctx, const SSElement& attr,
                                     const OperatorBase::in_list_t& ins,
                                     OperatorBase::out_list_t& outs) {
  int im2col_step;
  SSAttrs(attr)
      .get<int>("im2col_step", im2col_step)
      .done();
  const auto& value = ins[0];
  const auto& spatial_shapes = ins[1];
  const auto& level_start_index = ins[2];
  const auto& sampling_loc = ins[3];
  const auto& attn_weight = ins[4];
  const auto& grad_output = ins[5];


  auto out =
       ms_deform_attn_mlu_forward(ctx, value,
                                  spatial_shapes,
                                  level_start_index,
                                  sampling_loc,
                                  attn_weight,
                                  im2col_step);
  outs[0] = out;
  //copy(ctx, outs[0], out);

}


void ms_deform_attn_mlu_backward_camb_parrots(CambContext& ctx, const SSElement& attr,
                                     const OperatorBase::in_list_t& ins,
                                     OperatorBase::out_list_t& outs) {
  int im2col_step;
  SSAttrs(attr)
      .get<int>("im2col_step", im2col_step)
      .done();
  const auto& value = ins[0];
  const auto& spatial_shapes = ins[1];
  const auto& level_start_index = ins[2];
  const auto& sampling_loc = ins[3];
  const auto& attn_weight = ins[4];
  const auto& grad_output = ins[5];
  auto& grad_value = outs[0];
  auto& grad_sampling_loc = outs[1];
  auto& grad_attn_weight = outs[2];

  ms_deform_attn_mlu_backward(ctx, value,  spatial_shapes,
    level_start_index,  sampling_loc,
    attn_weight,grad_output, grad_value,
    grad_sampling_loc, grad_attn_weight,
    im2col_step);
}
					 

PARROTS_EXTENSION_REGISTER(ms_deform_attn_forward)
    .attr("im2col_step")
    .input(5)
    .output(1)
    .apply(ms_deform_attn_mlu_forward_camb_parrots)
    .done();		
		
PARROTS_EXTENSION_REGISTER(ms_deform_attn_backward)
    .attr("im2col_step")
    .input(6)
    .output(3)
    .apply(ms_deform_attn_mlu_backward_camb_parrots)
    .done();




