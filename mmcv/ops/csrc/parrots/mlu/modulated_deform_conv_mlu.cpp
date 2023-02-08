// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>
#include <parrots/darray/darraymath.hpp>

#ifdef PARROTS_USE_CAMB

#include <parrots/compute/cnnldescriptor.hpp>
#include <parrots/compute/cnnlhandle.hpp>
#include <parrots/compute/cnnlquantize.hpp>

#include "modulated_deform_conv_pytorch.h"
#include "parrots_mlu_helper.hpp"

#endif

using namespace parrots;

void modulated_deformable_im2col_camb(
        cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue, const cnrtDataType_t d_type,
        const void* data_im, const void* data_offset, const void* data_mask,
        const int batch_size, const int channels, const int height_im,
        const int width_im, const int height_col, const int width_col,
        const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
        const int stride_h, const int stride_w, const int dilation_h,
        const int dilation_w, const int deformable_group, void* data_col);

void modulated_deformable_col2im_camb(
        cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue, const cnrtDataType_t d_type,
        const void* data_col, const void* data_offset, const void* data_mask,
        const int batch_size, const int channels, const int height_im,
        const int width_im, const int height_col, const int width_col,
        const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
        const int stride_h, const int stride_w, const int dilation_h,
        const int dilation_w, const int deformable_group, void *grad_im);

void modulated_deformable_col2im_coord_camb(
        cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue, const cnrtDataType_t d_type,
        const void *data_col, const void *data_im, const void *data_offset,
        const void *data_mask, const int batch_size, const int channels,
        const int height_im, const int width_im, const int height_col,
        const int width_col, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w, const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w, const int deformable_group,
        void *grad_offset, void *grad_mask);

void modulated_deform_conv_forward_cpu_parrots(
        HostContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
        OperatorBase::out_list_t& outs) {
    int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
        dilation_w, group, deformable_group, with_bias;
    SSAttrs(attr)
        .get<int>("kernel_h", kernel_h)
        .get<int>("kernel_w", kernel_w)
        .get<int>("stride_h", stride_h)
        .get<int>("stride_w", stride_w)
        .get<int>("pad_h", pad_h)
        .get<int>("pad_w", pad_w)
        .get<int>("dilation_h", dilation_h)
        .get<int>("dilation_w", dilation_w)
        .get<int>("group", group)
        .get<int>("deformable_group", deformable_group)
        .get<int>("with_bias", with_bias)
        .done();

    const auto& input = buildATensor(ctx, ins[0]);
    const auto& weight = buildATensor(ctx, ins[1]);
    const auto& bias = buildATensor(ctx, ins[2]);
    const auto& ones = buildATensor(ctx, ins[3]);
    const auto& offset = buildATensor(ctx, ins[4]);
    const auto& mask = buildATensor(ctx, ins[5]);

    auto output = buildATensor(ctx, outs[0]);
    auto columns = buildATensor(ctx, outs[1]);

    modulated_deform_conv_forward(input, weight, bias, ones, offset, mask, output,
                                    columns, kernel_h, kernel_w, stride_h, stride_w,
                                    pad_h, pad_w, dilation_h, dilation_w, group,
                                    deformable_group, with_bias);
}

void modulated_deform_conv_backward_cpu_parrots(
        HostContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
        OperatorBase::out_list_t& outs) {
    int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
        dilation_w, group, deformable_group, with_bias;
    SSAttrs(attr)
        .get<int>("kernel_h", kernel_h)
        .get<int>("kernel_w", kernel_w)
        .get<int>("stride_h", stride_h)
        .get<int>("stride_w", stride_w)
        .get<int>("pad_h", pad_h)
        .get<int>("pad_w", pad_w)
        .get<int>("dilation_h", dilation_h)
        .get<int>("dilation_w", dilation_w)
        .get<int>("group", group)
        .get<int>("deformable_group", deformable_group)
        .get<int>("with_bias", with_bias)
        .done();

    const auto& input = buildATensor(ctx, ins[0]);
    const auto& weight = buildATensor(ctx, ins[1]);
    const auto& bias = buildATensor(ctx, ins[2]);
    const auto& ones = buildATensor(ctx, ins[3]);
    const auto& offset = buildATensor(ctx, ins[4]);
    const auto& mask = buildATensor(ctx, ins[5]);

    auto columns = buildATensor(ctx, outs[0]);
    auto grad_input = buildATensor(ctx, outs[1]);
    auto grad_weight = buildATensor(ctx, outs[2]);
    auto grad_bias = buildATensor(ctx, outs[3]);
    auto grad_offset = buildATensor(ctx, outs[4]);
    auto grad_mask = buildATensor(ctx, outs[5]);
    auto grad_output = buildATensor(ctx, outs[6]);
    modulated_deform_conv_backward(
        input, weight, bias, ones, offset, mask, columns, grad_input, grad_weight,
        grad_bias, grad_offset, grad_mask, grad_output, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
        deformable_group, with_bias);
}

#ifdef PARROTS_USE_CAMB

static void cambTransposeTo(ContextBase& ctx, const DArrayLite& in, DArrayLite& out, cnnlTensorLayout_t layoutIn,
                            cnnlTensorLayout_t layoutOut) {
    CnnlHandle& handle = CnnlHandle::get(ctx);
    CnnlTensorDesc inDesc(in.spec(), layoutIn);
    CnnlTensorDesc outDesc(in.spec(), layoutOut);
    std::vector<int> order;
    if (layoutIn == CNNL_LAYOUT_NHWC && layoutOut == CNNL_LAYOUT_HWCN) {
        order = {1, 2, 3, 0};
    } else if (layoutIn == CNNL_LAYOUT_NHWC && layoutOut == CNNL_LAYOUT_NCHW) {
        order = {0, 3, 1, 2};
    } else if (layoutIn == CNNL_LAYOUT_NCHW && layoutOut == CNNL_LAYOUT_HWCN) {
        order = {2, 3, 1, 0};
    } else if (layoutIn == CNNL_LAYOUT_NCHW && layoutOut == CNNL_LAYOUT_NHWC) {
        order = {0, 2, 3, 1};
    } else if (layoutIn == CNNL_LAYOUT_HWCN && layoutOut == CNNL_LAYOUT_NHWC) {
        order = {3, 0, 1, 2};
    } else if (layoutIn == CNNL_LAYOUT_HWCN && layoutOut == CNNL_LAYOUT_NCHW) {
        order = {3, 2, 0, 1};
    } else {
        PARROTS_NOTSUPPORTED << "cambTransposeTo requires "
                << "parrots::MemoryFormat in [CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_HWCN].\n";
    }

    CnnlTransposeDesc transDesc(4, order.data());
    PARROTS_CALLCNNL(
        cnnlTranspose(handle.native(), transDesc.get(), inDesc.get(), in.data(), outDesc.get(), out.data()));
}

void modulated_deform_conv_forward_camb_parrots(
        CambContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
        OperatorBase::out_list_t& outs) {
    int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
        dilation_w, group, deformable_group, with_bias;
    SSAttrs(attr)
        .get<int>("kernel_h", kernel_h)
        .get<int>("kernel_w", kernel_w)
        .get<int>("stride_h", stride_h)
        .get<int>("stride_w", stride_w)
        .get<int>("pad_h", pad_h)
        .get<int>("pad_w", pad_w)
        .get<int>("dilation_h", dilation_h)
        .get<int>("dilation_w", dilation_w)
        .get<int>("group", group)
        .get<int>("deformable_group", deformable_group)
        .get<int>("with_bias", with_bias)
        .done();

    DArrayLite input = ins[0];
    DArrayLite weight = ins[1];
    DArrayLite bias = ins[2];
    DArrayLite ones = ins[3];
    DArrayLite offset = ins[4];
    DArrayLite mask = ins[5];

    DArrayLite& output = outs[0];
    DArrayLite& columns = outs[1];

    const int batch = input.dim(0);
    const int channels = input.dim(1);
    const int height = input.dim(2);
    const int width = input.dim(3);

    const int channels_out = weight.dim(0);
    const int channels_kernel = weight.dim(1);
    const int kernel_h_ = weight.dim(2);
    const int kernel_w_ = weight.dim(3);

    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
        AT_ERROR("Input shape and kernel shape won't match: (%d x %d vs %d x %d).",
                kernel_h_, kernel_w, kernel_h_, kernel_w_);
    if (channels != channels_kernel * group)
        AT_ERROR("Input shape and kernel channels won't match: (%d vs %d).",
                channels, channels_kernel * group);

    const int height_out =
        (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out =
        (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (ones.ndims() != 2 ||
        ones.dim(0) * ones.dim(1) < height_out * width_out) {
        // Resize plane and fill with ones...
        ones = ctx.createDArrayLite(input.elemType(),
                                    DArrayShape(height_out, width_out));
        fill(ctx, ones, 1);
    }

    // resize output
    output = output.view({batch, channels_out, height_out, width_out});
    fill(ctx, output, 0);

    if (output.dim(2) == offset.dim(2) && output.dim(3) == offset.dim(3)) {
        std::vector<int> padding_t(4);
        std::vector<int> stride_t(2);
        std::vector<int> dilation_t(2);
        padding_t[0] = pad_h;
        padding_t[1] = pad_w;
        padding_t[2] = pad_h;
        padding_t[3] = pad_w;
        stride_t[0] = stride_h;
        stride_t[1] = stride_w;
        dilation_t[0] = dilation_h;
        dilation_t[1] = dilation_w;
        int im2col_step = 1;

        DArrayLite inputTemp, offsetTemp, maskTemp, weightTemp, outputTemp;
        inputTemp = ctx.createDArrayLite(input.spec().duplicate(parrots::MemoryFormat::ChannelsLast));
        offsetTemp = ctx.createDArrayLite(offset.spec().duplicate(parrots::MemoryFormat::ChannelsLast));
        maskTemp = ctx.createDArrayLite(mask.spec().duplicate(parrots::MemoryFormat::ChannelsLast));
        weightTemp = ctx.createDArrayLite(weight.spec().duplicate(parrots::MemoryFormat::ChannelsLast));
        outputTemp = ctx.createDArrayLite(output.spec().duplicate(parrots::MemoryFormat::ChannelsLast));
        cambTransposeTo(ctx, input, inputTemp, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
        cambTransposeTo(ctx, offset, offsetTemp, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
        cambTransposeTo(ctx, mask, maskTemp, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
        cambTransposeTo(ctx, weight, weightTemp, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
        cambTransposeTo(ctx, output, outputTemp, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);

        cnnlDCNDescriptor_t dcn_desc;
        PARROTS_CALLCNNL(cnnlCreateDCNDescriptor(&dcn_desc));
        PARROTS_CALLCNNL(cnnlSetDCNDescriptor(dcn_desc, inputTemp.ndims(), padding_t.data(), stride_t.data(),
                dilation_t.data(), deformable_group, group, im2col_step, CNNL_DTYPE_FLOAT));

        CnnlHandle& handle = CnnlHandle::get(ctx);
        CnnlTensorDesc inputDesc(inputTemp.spec());
        CnnlTensorDesc offsetDesc(offsetTemp.spec());
        CnnlTensorDesc maskDesc(maskTemp.spec());
        CnnlTensorDesc weightDesc(weightTemp.spec());
        CnnlTensorDesc biasDesc(bias.spec());
        CnnlTensorDesc outputDesc(outputTemp.spec());
        size_t workspace_size = 0;

        cnnlDataType_t onChipDataType = getQuantifyDtype(input.elemType());
        inputDesc.setOnchipDtype(onChipDataType);
        offsetDesc.setOnchipDtype(onChipDataType);
        weightDesc.setOnchipDtype(onChipDataType);
        outputDesc.setOnchipDtype(getCnnlDataType(outputTemp.elemType()));


        PARROTS_CALLCNNL(cnnlGetDCNForwardWorkspaceSize(handle.native(), dcn_desc, inputDesc.get(), offsetDesc.get(),
                maskDesc.get(), weightDesc.get(), biasDesc.get(), outputDesc.get(), &workspace_size));

        DArrayLite workspace = ctx.createDArrayLite(type_<char>(), DArrayShape(workspace_size));
        PARROTS_CALLCNNL(cnnlDCNForward(handle.native(), dcn_desc, inputDesc.get(), inputTemp.data(),
                offsetDesc.get(), offsetTemp.data(), maskDesc.get(), maskTemp.data(), weightDesc.get(),weightTemp.data(),
                with_bias ? biasDesc.get() : nullptr,
                with_bias ? bias.data() : nullptr,
                workspace.data(), workspace_size, outputDesc.get(), outputTemp.data()));
        PARROTS_CALLCNNL(cnnlDestroyDCNDescriptor(dcn_desc));

        cambTransposeTo(ctx, outputTemp, output, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);

    } else {
        columns = ctx.createDArrayLite(input.elemType(),
                                   DArrayShape(channels * kernel_h * kernel_w, 1 * height_out * width_out));
        fill(ctx, columns, 0);
        output = output.view({output.dim(0), group, output.dim(1) / group,
                                output.dim(2), output.dim(3)});

        DArrayLite input_nhwc = ctx.createDArrayLite(input.spec().duplicate(parrots::MemoryFormat::ChannelsLast));
        cambTransposeTo(ctx, input, input_nhwc, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
        columns = columns.view({1, channels, kernel_h * kernel_w, height_out * width_out});
        DArrayLite columns_nhwc = ctx.createDArrayLite(columns.spec().duplicate(parrots::MemoryFormat::ChannelsLast));
        for (int b = 0; b < batch; b++) {
            cnrtDim3_t k_dim = {getDeviceAttr(cnrtAttrMcorePerCluster), getDeviceAttr(cnrtAttrClusterCount), 1};
            cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
            auto queue = ctx.getStream().native();
            cnrtDataType_t d_type = getCnrtDataType(input.elemType());

            // launch kernel
            modulated_deformable_im2col_camb(
                k_dim, k_type, queue, d_type,
                input_nhwc[b].data(), offset[b].data(), mask[b].data(),
                1, channels, height,
                width, height_out, width_out,
                kernel_h, kernel_w, pad_h, pad_w,
                stride_h, stride_w, dilation_h,
                dilation_w, deformable_group, columns_nhwc.data());
            cambTransposeTo(ctx, columns_nhwc, columns, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);
            // divide into group
            weight = weight.view({group, weight.dim(0) / group, weight.dim(1),
                                weight.dim(2), weight.dim(3)});
            columns = columns.view({group, channels * kernel_h * kernel_w / group, 1 * height_out * width_out});

            for (int g = 0; g < group; g++) {
                DArrayLite outputBG = output[b][g].view({output[b][g].dim(0), output[b][g].size() / output[b][g].dim(0)});
                DArrayLite weightG = weight[g].view({weight[g].dim(0), weight[g].size() / weight[g].dim(0)});
                DArrayLite matOut = ctx.createDArrayLite(input.elemType(), DArrayShape(weightG.dim(0), columns[g].dim(1)));
                fill(ctx, matOut, 0);
                gemm(ctx, 1.0, false, weightG, false, columns[g], 0.0, matOut);
                add(ctx, outputBG, matOut, outputBG);
            }

            weight = weight.view({weight.dim(0) * weight.dim(1), weight.dim(2),
                                weight.dim(3), weight.dim(4)});
            columns = columns.view({columns.dim(0) * columns.dim(1), columns.dim(2)});
        }
        output = output.view({output.dim(0), output.dim(1) * output.dim(2),
                            output.dim(3), output.dim(4)});
        if (with_bias) {
            add(ctx, output, bias.view({1, bias.dim(0), 1, 1}), output);
        }
    }


}

void modulated_deform_conv_backward_camb_parrots(
        CambContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
        OperatorBase::out_list_t& outs) {
    int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
        dilation_w, group, deformable_group, with_bias;
    SSAttrs(attr)
        .get<int>("kernel_h", kernel_h)
        .get<int>("kernel_w", kernel_w)
        .get<int>("stride_h", stride_h)
        .get<int>("stride_w", stride_w)
        .get<int>("pad_h", pad_h)
        .get<int>("pad_w", pad_w)
        .get<int>("dilation_h", dilation_h)
        .get<int>("dilation_w", dilation_w)
        .get<int>("group", group)
        .get<int>("deformable_group", deformable_group)
        .get<int>("with_bias", with_bias)
        .done();

    DArrayLite input = ins[0];
    DArrayLite weight = ins[1];
    DArrayLite bias = ins[2];
    DArrayLite ones = ins[3];
    DArrayLite offset = ins[4];
    DArrayLite mask = ins[5];

    DArrayLite& columns = outs[0];
    DArrayLite& grad_input = outs[1];
    DArrayLite& grad_weight = outs[2];
    DArrayLite& grad_bias = outs[3];
    DArrayLite& grad_offset = outs[4];
    DArrayLite& grad_mask = outs[5];
    DArrayLite& grad_output = outs[6];

    const int batch = input.dim(0);
    const int channels = input.dim(1);
    const int height = input.dim(2);
    const int width = input.dim(3);

    const int channels_out = weight.dim(0);
    const int channels_kernel = weight.dim(1);
    const int kernel_h_ = weight.dim(2);
    const int kernel_w_ = weight.dim(3);

    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
        AT_ERROR("Input shape and kernel shape won't match: (%d x %d vs %d x %d).",
                kernel_h_, kernel_w, kernel_h_, kernel_w_);
    if (channels != channels_kernel * group)
        AT_ERROR("Input shape and kernel channels won't match: (%d vs %d).",
                channels, channels_kernel * group);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (ones.ndims() != 2 || ones.dim(0) * ones.dim(1) < height_out * width_out) {
        // Resize plane and fill with ones...
        ones = ctx.createDArrayLite(input.elemType(), DArrayShape(height_out, width_out));
        fill(ctx, ones, 1);
    }
    grad_input = grad_input.view({batch, channels, height, width});
    columns = ctx.createDArrayLite(input.elemType(),
                                   DArrayShape(channels * kernel_h * kernel_w, height_out * width_out));
    fill(ctx, columns, 0);
    grad_output = grad_output.view({grad_output.dim(0), group, grad_output.dim(1) / group,
                            grad_output.dim(2), grad_output.dim(3)});
    DArrayLite input_nhwc = ctx.createDArrayLite(input.spec().duplicate(parrots::MemoryFormat::ChannelsLast));
    cambTransposeTo(ctx, input, input_nhwc, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);

    DArrayLite grad_input_nhwc = ctx.createDArrayLite(grad_input.spec().duplicate(parrots::MemoryFormat::ChannelsLast));
    cambTransposeTo(ctx, grad_input, grad_input_nhwc, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);

    for (int b = 0; b < batch; b++) {
        cnrtDim3_t k_dim = {getDeviceAttr(cnrtAttrMcorePerCluster), getDeviceAttr(cnrtAttrClusterCount), 1};
        cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
        auto queue = ctx.getStream().native();
        cnrtDataType_t d_type = getCnrtDataType(input.elemType());
        // divide int group
        columns = columns.view({group, columns.dim(0) / group, columns.dim(1)});
        weight = weight.view({group, weight.dim(0) / group, weight.dim(1),
                            weight.dim(2), weight.dim(3)});
        for (int g = 0; g < group; g++) {
            DArrayLite weightG = weight[g].view({weight[g].dim(0), weight[g].size() / weight[g].dim(0)});
            weightG = transpose(ctx, weightG, 0, 1);
            DArrayLite grad_outputBG = grad_output[b][g].view({grad_output[b][g].dim(0),
                                                               grad_output[b][g].size() / grad_output[b][g].dim(0)});
            gemm(ctx, 1.0, false, weightG, false, grad_outputBG, 0.0, columns[g]);
        }

        columns = columns.view({columns.dim(0) * columns.dim(1), columns.dim(2)});
        weight = weight.view({weight.dim(0) * weight.dim(1), weight.dim(2), weight.dim(3), weight.dim(4)});

        DArrayLite columns_nchw = columns.view({1, channels, kernel_h * kernel_w, height_out * width_out});
        DArrayLite columns_nhwc = ctx.createDArrayLite(columns_nchw.spec().duplicate(parrots::MemoryFormat::ChannelsLast));
        cambTransposeTo(ctx, columns_nchw, columns_nhwc, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);

        // gradient w.r.t. input coordinate data
        modulated_deformable_col2im_coord_camb(
            k_dim, k_type, queue, d_type,
            columns_nhwc.data(), input_nhwc[b].data(), offset[b].data(), mask[b].data(),
            1, channels, height, width,
            height_out, width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h,
            stride_w, dilation_h, dilation_w, deformable_group, grad_offset[b].data(),
            grad_mask[b].data());

        // gradient w.r.t. input data
        modulated_deformable_col2im_camb(
            k_dim, k_type, queue, d_type,
            columns_nhwc.data(), offset[b].data(), mask[b].data(),
            1, channels, height, width, height_out,
            width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, deformable_group, grad_input_nhwc[b].data());

        // gradient w.r.t. weight, dWeight should accumulate across the batch and
        // group
        modulated_deformable_im2col_camb(
            k_dim, k_type, queue, d_type,
            input_nhwc[b].data(), offset[b].data(), mask[b].data(),
            1, channels, height, width, height_out,
            width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, deformable_group, columns_nhwc.data());
        cambTransposeTo(ctx, columns_nhwc, columns_nchw, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);

        columns = columns_nchw.view({group, columns.dim(0) / group, columns.dim(1)});
        grad_weight = grad_weight.view({group, grad_weight.dim(0) / group,
                                        grad_weight.dim(1), grad_weight.dim(2),
                                        grad_weight.dim(3)});
        if (with_bias) {
            grad_bias = grad_bias.view({group, grad_bias.dim(0) / group});
        }

        for (int g = 0; g < group; g++) {
            DArrayLite grad_outputBG = grad_output[b][g].view({grad_output[b][g].dim(0),
                                                            grad_output[b][g].size() / grad_output[b][g].dim(0)});
            DArrayLite columnsG = transpose(ctx, columns[g], 0, 1);
            DArrayLite matOut = ctx.createDArrayLite(input.elemType(), DArrayShape(grad_outputBG.dim(0), columnsG.dim(1)));
            fill(ctx, matOut, 0);
            gemm(ctx, 1.0, false, grad_outputBG, false, columnsG, 0.0, matOut);
            DArrayLite grad_weightG = grad_weight[g].view({grad_weight[g].dim(0), grad_weight[g].size() / grad_weight[g].dim(0)});
            add(ctx, grad_weightG, matOut, grad_weightG);
            if (with_bias) {
                DArrayLite grad_weightG = grad_bias[g].view({grad_bias[g].size(), 1});
                ones = ones.view({ones.size(), 1});
                DArrayLite matOut = ctx.createDArrayLite(input.elemType(), DArrayShape(grad_outputBG.dim(0), ones.dim(1)));
                fill(ctx, matOut, 0);
                gemm(ctx, 1.0, false, grad_outputBG, false, ones, 0.0, matOut);
                add(ctx, grad_weightG, matOut, grad_weightG);
            }
        }

        columns = columns.view({columns.dim(0) * columns.dim(1), columns.dim(2)});
        grad_weight = grad_weight.view({grad_weight.dim(0) * grad_weight.dim(1),
                                        grad_weight.dim(2), grad_weight.dim(3),
                                        grad_weight.dim(4)});
        if (with_bias) {
            grad_bias = grad_bias.view({grad_bias.dim(0) * grad_bias.dim(1)});
        }
    }
    cambTransposeTo(ctx, grad_input_nhwc, grad_input, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);

    grad_output = grad_output.view({grad_output.dim(0) * grad_output.dim(1),
                            grad_output.dim(2), grad_output.dim(3),
                            grad_output.dim(4)});
}

#endif  // PARROTS_USE_CAMB

void modulated_deform_conv_forward_parrots(
        HostContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
        OperatorBase::out_list_t& outs) {
    if (ctx.getProxy().isHost()) {
        modulated_deform_conv_forward_cpu_parrots(ctx, attr, ins, outs);
    } else {
#ifdef PARROTS_USE_CAMB
        modulated_deform_conv_forward_camb_parrots(ctx, attr, ins, outs);
#endif
    }
}

void modulated_deform_conv_backward_parrots(
        HostContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
        OperatorBase::out_list_t& outs) {
    if (ctx.getProxy().isHost()) {
        modulated_deform_conv_backward_cpu_parrots(ctx, attr, ins, outs);
    } else {
#ifdef PARROTS_USE_CAMB
        modulated_deform_conv_backward_camb_parrots(ctx, attr, ins, outs);
#endif
    }
}


PARROTS_EXTENSION_REGISTER(modulated_deform_conv_forward)
    .attr("kernel_h")
    .attr("kernel_w")
    .attr("stride_h")
    .attr("stride_w")
    .attr("pad_h")
    .attr("pad_w")
    .attr("dilation_h")
    .attr("dilation_w")
    .attr("group")
    .attr("deformable_group")
    .attr("with_bias")
    .input(6)
    .output(2)
    .apply(modulated_deform_conv_forward_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(modulated_deform_conv_backward)
    .attr("kernel_h")
    .attr("kernel_w")
    .attr("stride_h")
    .attr("stride_w")
    .attr("pad_h")
    .attr("pad_w")
    .attr("dilation_h")
    .attr("dilation_w")
    .attr("group")
    .attr("deformable_group")
    .attr("with_bias")
    .input(6)
    .output(7)
    .apply(modulated_deform_conv_backward_parrots)
    .done();


