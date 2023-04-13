
#define MLUOP_MAJOR 0
#define MLUOP_MINOR 4
#define MLUOP_PATCHLEVEL 2

#include <vector>

#include "parrots_mlu_helper.hpp"
#include "mlu_op.h"

using namespace parrots;

mluOpDataType_t getMluOpDataType(parrots::ValueType vt);
mluOpTensorLayout_t getMluOpSuggestLayout(const DArrayLite& input);

class MluOpTensorDescriptor {
 public:
  MluOpTensorDescriptor() { mluOpCreateTensorDescriptor(&desc_); };
  ~MluOpTensorDescriptor() { mluOpDestroyTensorDescriptor(desc_); }

  void set(DArrayLite);
  mluOpTensorDescriptor_t desc() { return desc_; }

 private:
  mluOpTensorDescriptor_t desc_;
  void set_desc(const DArrayLite&, mluOpTensorLayout_t, mluOpDataType_t,
                std::vector<int>& dims);
};

mluOpHandle_t mluOpGetCurrentHandle(CambContext& ctx);

class MluOpHandle {
 public:
  MluOpHandle() : handle(nullptr) { mluOpCreate(&handle); }
  ~MluOpHandle() {
    if (handle) {
      mluOpDestroy(handle);
      handle = nullptr;
    }
  }
  void setQueue(cnrtQueue_t queue) { mluOpSetQueue(handle, queue); }
  mluOpHandle_t handle;
};
