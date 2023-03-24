#ifndef VOXELIZATION_UTILS_HPP_
#define VOXELIZATION_UTILS_HPP_

/*!
 * @brief Converts int32 to float32 data type.
 *
 * @param[out] dst
 *   Pointer to NRAM that stores int32 type data.
 * @param[in,out] dst_addition
 *   Pointer to NRAM as the workspace of dst, which has the same size as dst.
 *   It allows empty pointer on MLU300 series.
 * @param[in] src
 *   Pointer to NRAM that stores float32 type data.
 * @param[in,out] src_addition
 *   Pointer to NRAM as the workspace of src, which has a size of 128 Bytes.
 *   It allows empty pointer on MLU300 series.
 * @param[in] src_count
 *   The count of elements in src.
 */
__mlu_func__ void convertInt2Float(float *dst, float *dst_addition, int *src,
                                   float *src_addition, const int src_count) {
#if __BANG_ARCH__ >= 300
  __bang_int2float((float *)dst, (int32_t *)src, src_count, 0);
#else
  // get sign bit
  const float move_23bit = 8388608.0;
  // 0x80000000 = 1,000000000,0000000000000000000000000000
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0x80000000);
  __bang_cycle_band((char *)dst_addition, (char *)src, (char *)src_addition,
                    src_count * sizeof(float), NFU_ALIGN_SIZE);
  // get 1 or 0 from sign bit
  // judg is Odd
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0x00000001);
  __bang_cycle_bor((char *)dst_addition, (char *)dst_addition,
                   (char *)src_addition, src_count * sizeof(float),
                   NFU_ALIGN_SIZE);
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0x80000001);
  __bang_cycle_eq(dst_addition, dst_addition, src_addition, src_count,
                  NFU_ALIGN_SIZE / sizeof(float));
  // minus xor, positive num invariant
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0xffffffff);
  __bang_cycle_mul(dst, dst_addition, src_addition, src_count,
                   NFU_ALIGN_SIZE / sizeof(float));
  __bang_bxor((char *)dst, (char *)src, (char *)dst, src_count * sizeof(float));
  // convert int32 to float32
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0x7fffff);
  __bang_cycle_band((char *)dst, (char *)dst, (char *)src_addition,
                    src_count * sizeof(float), NFU_ALIGN_SIZE);
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0x4b000000);
  __bang_cycle_bor((char *)dst, (char *)dst, (char *)src_addition,
                   src_count * sizeof(float), NFU_ALIGN_SIZE);
  __bang_sub_scalar(dst, dst, move_23bit, src_count);
  // add one
  __bang_add(dst, dst, dst_addition, src_count);
  // set sign for float32
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0xffffffff);
  __bang_cycle_mul(dst_addition, dst_addition, src_addition, src_count,
                   NFU_ALIGN_SIZE / sizeof(float));

  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0x00000001);
  __bang_cycle_add(dst_addition, dst_addition, src_addition, src_count,
                   NFU_ALIGN_SIZE / sizeof(float));

  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0x80000000);
  __bang_cycle_band((char *)dst_addition, (char *)dst_addition,
                    (char *)src_addition, src_count * 4, 128);
  __bang_bor((char *)dst, (char *)dst, (char *)dst_addition, src_count * 4);
#endif  // __BANG_ARCH__ >= 300
}

/*!
 * @brief Converts float32 to int32 data type with to_zero round mode.
 *
 * @param[out] dst
 *   Pointer to NRAM that stores float32 type data.
 * @param[in,out] dst_addition
 *   Pointer to NRAM as the workspace of dst, which has the same size as dst.
 *   It allows empty pointer on MLU300 series.
 * @param[in] src
 *   Pointer to NRAM that stores int32 type data.
 * @param[in,out] src_addition
 *   Pointer to NRAM as the workspace of src, which has a size of 128 Bytes.
 *   It allows empty pointer on MLU300 series.
 * @param[in] src_count
 *   The count of elements in src.
 */
__mlu_func__ void convertFloat2Int(int *dst, float *dst_addition, float *src,
                                   float *src_addition, const int src_count) {
#if __BANG_ARCH__ >= 300
  __bang_float2int_tz((int32_t *)dst, (float *)src, src_count, 0);
#else
  // sign ===> src_addition
  // dst=-1.0 : when src[i] is a negative number
  // dst=+1.0 : when src[i] is a positive number
  const int floatDchar = sizeof(float) / sizeof(char);
  __bang_active_sign((float *)dst, src, src_count);
  // dst_addition = abs(src)
  __bang_mul(dst_addition, src, (float *)dst, src_count);
  // if dst_addition < 1.0 , then src_addition + 1, to fix add error.
  __bang_write_value((float *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     1.0f);
  __bang_cycle_lt(dst_addition, dst_addition, (float *)src_addition, src_count,
                  NFU_ALIGN_SIZE / sizeof(float));
  __bang_add_tz((float *)dst, (float *)dst, (float *)dst_addition, src_count);
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0xbf800000);
  // set negative flag -1.0 = 0xbf80000
  __bang_cycle_eq(
      (float *)dst, (float *)dst, (float *)src_addition, src_count,
      NFU_ALIGN_SIZE / sizeof(float));  //  to mark all src in [x<-1.0]
  __bang_active_abs(dst_addition, src, src_count);
  __bang_write_value((float *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     8388608.0f);
  // mask shift move 23
  __bang_cycle_add_tz(
      dst_addition, dst_addition, src_addition, src_count,
      NFU_ALIGN_SIZE / sizeof(float));  // right shift move 23bit
  // two`s complement for negatibe
  // dst=1.0 , when src <-1.0
  // dst=0.0 , when src >=-1.0
  __bang_sub(dst_addition, dst_addition, (float *)dst, src_count);
  // to fix max value
  // 0 1001 0110 111 1111 1111 1111 1111 1111 <=> 0xcb7fffff <=> 16777215.0,
  // means max value.
  __bang_mul_scalar((float *)dst, (float *)dst, 16777215.0, src_count);
  __bang_bxor((char *)dst_addition, (char *)dst_addition, (char *)dst,
              src_count * floatDchar);
  // get low 23bit
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     (unsigned)0x007fffff);
  // mask low 23bit is 1
  __bang_cycle_band((char *)dst_addition, (char *)dst_addition,
                    (char *)src_addition, src_count * floatDchar,
                    NFU_ALIGN_SIZE / sizeof(char));
  // set 9 high bit ===> dst
  // -2.0 <=> 0xc0000000 <=> 1100 0000 0000 0000 0000 0000 0000 0000
  //  1.0 <=> 0x3f800000 <=> 0011 1111 1000 0000 0000 0000 0000 0000
  __bang_write_value(src_addition, NFU_ALIGN_SIZE / sizeof(float), 0x3f800000);
  __bang_cycle_and((float *)dst, (float *)dst, src_addition, src_count,
                   NFU_ALIGN_SIZE / sizeof(float));
  // src or dst_addition
  __bang_bor((char *)dst_addition, (char *)dst, (char *)dst_addition,
             src_count * floatDchar);
  __bang_mul_scalar((float *)dst, (float *)dst, -2.0, src_count);
  __bang_bor((char *)dst, (char *)dst, (char *)dst_addition,
             src_count * floatDchar);
#endif  // __BANG_ARCH__ >= 300
}

#endif  // 