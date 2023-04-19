#include <parrots/darray/darraymath.hpp>
#include <parrots_mlu_helper.hpp>

using namespace parrots;

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

void KernelDynamicVoxelize(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *points, void *coors, const float voxel_x, const float voxel_y,
    const float voxel_z, const float coors_x_min, const float coors_y_min,
    const float coors_z_min, const float coors_x_max, const float coors_y_max,
    const float coors_z_max, const int32_t grid_x, const int32_t grid_y,
    const int32_t grid_z, const int32_t num_points, const int32_t num_features);

void KernelPoint2Voxel(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                       cnrtQueue_t queue, void *coors, void *point_to_pointidx,
                       void *point_to_voxelidx, const int32_t num_points,
                       const int32_t max_points);

void KernelCalcPointsPerVoxel(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                              cnrtQueue_t queue, void *point_to_pointidx,
                              void *point_to_voxelidx, void *coor_to_voxelidx,
                              void *num_points_per_voxel, void *voxel_num,
                              const int32_t max_voxels,
                              const int32_t num_points);

void KernelAssignVoxelsCoors(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                             cnrtQueue_t queue, const void *points,
                             void *temp_coors, void *point_to_voxelidx,
                             void *coor_to_voxelidx, void *voxels, void *coors,
                             const int32_t max_points, const int32_t num_points,
                             const int32_t num_features);

// policy function
static void policyFuncDefault(cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type,
                              const int num_points) {
  k_dim->x = getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim->y = MIN((num_points + k_dim->x - 1) / k_dim->x,
                 getDeviceAttr(cnrtAttrClusterCount));
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
}

// policy function
static void policyFuncCalcPointsPerVoxel(cnrtDim3_t *k_dim,
                                         cnrtFunctionType_t *k_type,
                                         const int num_points) {
  k_dim->x = 1;
  k_dim->y = 1;
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_BLOCK;
}

void HardVoxelizeForwardMLUKernelLauncher(CambContext &ctx,
    const DArrayLite &points, DArrayLite &voxels, DArrayLite &coors,
    DArrayLite &num_points_per_voxel, DArrayLite &voxel_num,
    const std::vector<float> voxel_size, const std::vector<float> coors_range,
    const int max_points, const int max_voxels, const int NDim = 3) {
    // check datatype
    PARROTS_CHECKARGS(points.elemType() == Prim::Float32)
                << "points type should be Float, got " << points.elemType() << ".";
    PARROTS_CHECKARGS(voxels.elemType() == Prim::Float32)
                << "voxels type should be Float, got " << voxels.elemType() << ".";
    PARROTS_CHECKARGS(coors.elemType() == Prim::Int32)
                << "coors type should be Float, got " << coors.elemType() << ".";
    PARROTS_CHECKARGS(num_points_per_voxel.elemType() == Prim::Int32)
                << "num_points_per_voxel type should be Float, got "
                << num_points_per_voxel.elemType() << ".";

    // check shape
    PARROTS_CHECKARGS(points.ndims() == 2) << "points should be a 2d tensor, got "
                << points.ndims() << "D.";
    PARROTS_CHECKARGS(voxels.ndims() == 3) << "voxels should be a 3d tensor, got "
                << voxels.ndims() << "D.";
    PARROTS_CHECKARGS(coors.ndims() == 2) << "coors should be a 2d tensor, got "
                << coors.ndims() << "D.";
    PARROTS_CHECKARGS(num_points_per_voxel.ndims() == 1)
                << "num_points_per_voxel should be a 1d tensor, got "
                << num_points_per_voxel.ndims() << "D.";

    const int num_points = points.dim(0);
    const int num_features = points.dim(1);

    PARROTS_CHECKARGS(points.dim(0) == num_points)
                << "the 1st dimensions of points should be num_points, got "
                << points.dim(0) << ".";
    PARROTS_CHECKARGS(points.dim(1) == num_features)
                << "the 2nd dimensions of points should be num_features, got "
                << points.dim(1) << ".";
    PARROTS_CHECKARGS(voxels.dim(0) == max_voxels)
                << "the 1st dimensions of voxels should be max_voxels, got "
                << voxels.dim(0) << ".";
    PARROTS_CHECKARGS(voxels.dim(1) == max_points)
                << "the 2nd dimensions of voxels should be max_points, got "
                << voxels.dim(1) << ".";
    PARROTS_CHECKARGS(voxels.dim(2) == num_features)
                << "the 3rd dimensions of voxels should be num_features, got "
                << voxels.dim(2) << ".";
    PARROTS_CHECKARGS(coors.dim(0) == max_voxels)
                << "the 1st dimensions of coors should be max_voxels, got "
                << coors.dim(0) << ".";
    PARROTS_CHECKARGS(coors.dim(1) == 3)
                << "the 2nd dimensions of coors should be 3, got " << coors.dim(1)
                << ".";
    PARROTS_CHECKARGS(num_points_per_voxel.dim(0) == max_voxels)
                << "the 1st dimensions of num_points_per_voxel should be 3, got "
                << num_points_per_voxel.dim(0) << ".";

    // large tensor check
    const size_t max_input_size = 2147483648;
    PARROTS_CHECKARGS(points.size() < max_input_size)
                << "points element num should be less than 2^31, got "
                << points.size() << ".";
    PARROTS_CHECKARGS(voxels.size() < max_input_size)
                << "voxels element num should be less than 2^31, got "
                << voxels.size() << ".";
    PARROTS_CHECKARGS(coors.size() < max_input_size)
                << "coors element num should be less than 2^31, got " << coors.size()
                << ".";

    // check zero element
    if (max_points == 0 || max_voxels == 0) {
        return ;
    }

    // get compute queue
    auto queue = ctx.getStream().native();

    // get ptr of tensors
    auto points_ptr = points.data();
    auto voxels_ptr = voxels.data();
    auto coors_ptr = coors.data();
    auto num_points_per_voxel_ptr = num_points_per_voxel.data();

    // calculate task dimension
    cnrtDim3_t k_dim;
    cnrtFunctionType_t k_type;
    policyFuncDefault(&k_dim, &k_type, num_points);

    // 1. link point to corresponding voxel coors
    const float voxel_x = voxel_size[0];
    const float voxel_y = voxel_size[1];
    const float voxel_z = voxel_size[2];
    const float coors_x_min = coors_range[0];
    const float coors_y_min = coors_range[1];
    const float coors_z_min = coors_range[2];
    const float coors_x_max = coors_range[3];
    const float coors_y_max = coors_range[4];
    const float coors_z_max = coors_range[5];

    const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
    const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
    const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

    DArrayLite temp_coors = ctx.createDArrayLite(points.spec().withShape(
                    DArrayShape(NDim, num_points)).withElemType(Prim::Int32));
    auto temp_coors_ptr = temp_coors.data();
    fill(ctx, temp_coors, 0);

    KernelDynamicVoxelize(k_dim, k_type, queue, points_ptr, temp_coors_ptr,
                            voxel_x, voxel_y, voxel_z, coors_x_min, coors_y_min,
                            coors_z_min, coors_x_max, coors_y_max, coors_z_max,
                            grid_x, grid_y, grid_z, num_points, num_features);

    // 2. map point to the idx of the corresponding voxel, find duplicate coor

    DArrayLite point_to_pointidx = ctx.createDArrayLite(points.spec().withShape(
                    DArrayShape(num_points)).withElemType(Prim::Int32));
    auto point_to_pointidx_ptr = point_to_pointidx.data();
    fill(ctx, point_to_pointidx, 0);

    DArrayLite point_to_voxelidx = ctx.createDArrayLite(points.spec().withShape(
                    DArrayShape(num_points)).withElemType(Prim::Int32));
    auto point_to_voxelidx_ptr = point_to_voxelidx.data();
    fill(ctx, point_to_voxelidx, 0);

    KernelPoint2Voxel(k_dim, k_type, queue, temp_coors_ptr, point_to_pointidx_ptr,
                        point_to_voxelidx_ptr, num_points, max_points);

    // calculate task dimension
    cnrtDim3_t k_dim_calc_points_per_voxel;
    cnrtFunctionType_t k_type_calc_points_per_voxel;
    policyFuncCalcPointsPerVoxel(&k_dim_calc_points_per_voxel,
                                &k_type_calc_points_per_voxel, num_points);

    // 3. determine voxel num and voxel's coor index
    DArrayLite coor_to_voxelidx = ctx.createDArrayLite(points.spec().withShape(
                    DArrayShape(num_points)).withElemType(Prim::Int32));
    auto coor_to_voxelidx_ptr = coor_to_voxelidx.data();
    fill(ctx, coor_to_voxelidx, 0);

    DArrayLite voxel_num_tmp = ctx.createDArrayLite(points.spec().withShape(
                    DArrayShape(1)).withElemType(Prim::Int64));
    auto voxel_num_ptr = voxel_num_tmp.data();
    fill(ctx, voxel_num_tmp, 0);

    KernelCalcPointsPerVoxel(
        k_dim_calc_points_per_voxel, k_type_calc_points_per_voxel, queue,
        point_to_pointidx_ptr, point_to_voxelidx_ptr, coor_to_voxelidx_ptr,
        num_points_per_voxel_ptr, voxel_num_ptr, max_voxels, num_points);

    // 4. copy point features and coors of each voxels to voxels
    KernelAssignVoxelsCoors(k_dim, k_type, queue, points_ptr, temp_coors_ptr,
                            point_to_voxelidx_ptr, coor_to_voxelidx_ptr,
                            voxels_ptr, coors_ptr, max_points, num_points,
                            num_features);

    copy(ctx, voxel_num, voxel_num_tmp);
}

void hard_voxelize_forward_mlu_parrots(CambContext& ctx, const SSElement& attr,
                                       const OperatorBase::in_list_t& ins,
                                       OperatorBase::out_list_t& outs) {
    int max_points, max_voxels, NDim;
    bool deterministic;
    SSAttrs(attr)
        .get<int>("max_points", max_points)
        .get<int>("max_voxels", max_voxels)
        .get<int>("NDim", NDim)
        .get<bool>("deterministic", deterministic)
        .done();
    const auto& points = ins[0];
    // const auto& voxel_size = ins[1];
    // const auto& coors_range = ins[2];
    auto voxel_size_array = ins[1];
    auto coors_range_array = ins[2];
    std::vector<float> voxel_size(
        voxel_size_array.ptr<float>(),
        voxel_size_array.ptr<float>() + voxel_size_array.size()
    );
    std::vector<float> coors_range(
        coors_range_array.ptr<float>(),
        coors_range_array.ptr<float>() + coors_range_array.size()
    );

    DArrayLite& voxels = outs[0];
    DArrayLite& coors = outs[1];
    DArrayLite& num_points_per_voxel = outs[2];
    DArrayLite& voxel_num = outs[3];

    HardVoxelizeForwardMLUKernelLauncher(ctx, points, voxels, coors,
        num_points_per_voxel, voxel_num, voxel_size, coors_range, max_points, max_voxels, NDim);
}

PARROTS_EXTENSION_REGISTER(hard_voxelize_forward)
    .attr("max_points")
    .attr("max_voxels")
    .attr("NDim")
    .attr("deterministic")
    .input(3)
    .output(4)
    .apply(hard_voxelize_forward_mlu_parrots)
    .done();
