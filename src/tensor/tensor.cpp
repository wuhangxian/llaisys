#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    // TO_BE_IMPLEMENTED();
    // isContiguous 的判断依据是 stride 数组是否是单调递减的
    // 忽略点：考虑切片，也就是可能 offset 可能并不是 0 
    int dims = ndim();
    const auto& shapes = _meta.shape; 
    const auto& strides = _meta.strides;

    ptrdiff_t z = 1;

    for (int i = dims - 1; i >= 0; i--) {
        if (shapes[i] == 1) {
             continue; 
        }

        if (strides[i] != z) {
            return false;
        }

        z *= shapes[i];
    }

    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // TO_BE_IMPLEMENTED();
    if (order.size() != this->ndim()) {
        printf("Permute dimensions must match tensor dimensions");
        return NULL;
    }

    TensorMeta new_meta = _meta;
    
    new_meta.shape.resize(order.size());
    new_meta.strides.resize(order.size());

    for (size_t i = 0; i < order.size(); ++i) {
        // order[i] 是旧 Tensor 的哪个维度应该放到新的第 i 维
        size_t old_dim_index = order[i];

        new_meta.shape[i]   = _meta.shape[old_dim_index];
        new_meta.strides[i] = _meta.strides[old_dim_index];
    }

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // TO_BE_IMPLEMENTED();
    // 1.要求连续
    if (!this->isContiguous()) {
        printf("Error: view tensor must contiguous\n");
        return NULL;
    }

    // 2.要求元素数量一样多
    size_t new_numel = 1;
    for (auto s : shape) {
        new_numel *= s;
    }
    if (new_numel != this->numel()) {
        printf("Error: view tensor must contiguous\n");
        return NULL;
    }

    // 3.构造新的 strides 
    std::vector<ptrdiff_t> new_strides(shape.size());
    ptrdiff_t accumulated_stride = 1;
    
    // 从最后一个维度向前倒推
    for (int i = shape.size() - 1; i >= 0; --i) {
        new_strides[i] = accumulated_stride;
        accumulated_stride *= shape[i];
    }

    // 4. 构造元数据
    TensorMeta new_meta;
    new_meta.shape = shape;
    new_meta.strides = new_strides;
    new_meta.dtype = _meta.dtype; // 保持数据类型不变

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // TO_BE_IMPLEMENTED();
    // 1. 【安全检查】
    if (dim >= this->ndim()) {
        printf("Tensor::slice: Dimension out of range");
        return NULL;
    }
    // 检查索引边界
    if (start >= end) {
        printf("Tensor::slice: start must be less than end");
        return NULL;
    }
    if (end > _meta.shape[dim]) {
        printf("Tensor::slice: end index larger than dimension size");
        return NULL;
    }

    TensorMeta new_meta = _meta;

    // 1. 改变 shape， dim位置的维度变为 end - start
    new_meta.shape[dim] = end - start;

    size_t elem_size = 0;
    // offset 不是元素数目，而是字节
    switch (_meta.dtype) {
        case LLAISYS_DTYPE_BYTE:
        case LLAISYS_DTYPE_I8:
        case LLAISYS_DTYPE_BOOL:
        case LLAISYS_DTYPE_U8:
            elem_size = 1;
            break;

        case LLAISYS_DTYPE_I16:
        case LLAISYS_DTYPE_BF16:
        case LLAISYS_DTYPE_U16:
        case LLAISYS_DTYPE_F16:
            elem_size = 2;
            break;

        case LLAISYS_DTYPE_I32:
        case LLAISYS_DTYPE_U32:
        case LLAISYS_DTYPE_F32:
            elem_size = 4;
            break;

        case LLAISYS_DTYPE_I64:
        case LLAISYS_DTYPE_U64:
        case LLAISYS_DTYPE_F64:
            elem_size = 8;
            break;

        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(_meta.dtype);
    }

    // 2. offset: 跳过 start * stride[dim] 个元素，再乘以elem_size就是跳过多少个字节
    size_t shift_bytes = start * _meta.strides[dim] * elem_size;
    
    // 加上旧的 offset (支持多次连续切片)
    size_t new_offset = _offset + shift_bytes;

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    // TO_BE_IMPLEMENTED();
    if (!this->isContiguous()) {
         // 在实际框架中，这里应该抛出异常或进行处理
         printf("Error: Cannot load into non-contiguous tensor!\n");
         return;
    }

    size_t total_bytes = this->numel() * this->elementSize();
    void* dst_ptr = this->data();

    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(dst_ptr, src_, total_bytes);
    } 
    else if (this->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        // TODO(): cudaMemcpy(dst_ptr, src_, total_bytes, cudaMemcpyHostToDevice);
    }
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
