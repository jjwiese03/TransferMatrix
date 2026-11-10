#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

#include <cmath>
#include <cstdint>
#include <limits>

extern "C" {
struct data {
    cuDoubleComplex* P;
    int Pn0, Pn1, Pn2, Pn3;

    int64_t ML;
    const cuDoubleComplex* modes;
    int n0, n1, n2, n3;

    const double* R;
    int R_rows, R_cols;
    const double* kR;
    int kR_rows, kR_cols;
    const double* Phi;
    int Phi_rows, Phi_cols;

    const bool* diskmaskin;
    const bool* diskmaskout;
    int mask_rows, mask_cols;
};
}

namespace {
constexpr int blockSize = 256;
constexpr double speedOfLight = 299792458.0;

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : ptr_(nullptr) {
        status_ = cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T));
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }
    cudaError_t status() const { return status_; }

private:
    T* ptr_;
    cudaError_t status_;
};

__global__ void applySpectrum(cuDoubleComplex* fields, const double* kR,
                              size_t pixelCount, int modeCount,
                              double k0, double distance) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = pixelCount * modeCount;
    if (index >= total) {
        return;
    }

    const size_t pixel = index % pixelCount;
    const double radialK2 = kR[pixel];
    const cuDoubleComplex root =
        cuCsqrt(make_cuDoubleComplex(k0 * k0 - radialK2, 0.0));
    const cuDoubleComplex exponent =
        cuCmul(make_cuDoubleComplex(0.0, -distance), cuConj(root));
    fields[index] = cuCmul(fields[index], cuCexp(exponent));
}

__global__ void applyTiltAndMask(cuDoubleComplex* fields, const double* R,
                                 const double* Phi, const bool* mask,
                                 size_t pixelCount, int modeCount, double k0,
                                 double tiltx, double tilty,
                                 double inversePixelCount) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = pixelCount * modeCount;
    if (index >= total) {
        return;
    }

    const size_t pixel = index % pixelCount;
    if (!mask[pixel]) {
        fields[index] = make_cuDoubleComplex(0.0, 0.0);
        return;
    }

    const double phase = -k0 * (tiltx * R[pixel] * cos(Phi[pixel]) +
                                tilty * R[pixel] * sin(Phi[pixel]));
    const cuDoubleComplex tilt =
        make_cuDoubleComplex(cos(phase), sin(phase));
    fields[index] = cuCmul(fields[index], tilt);
    fields[index] = cuCmul(fields[index],
                           make_cuDoubleComplex(inversePixelCount, 0.0));
}

__global__ void reduceProjections(const cuDoubleComplex* fields,
                                  const cuDoubleComplex* modes,
                                  size_t pixelCount, int modeCount,
                                  int chunkCount, double* partials) {
    __shared__ double sums[3 * blockSize];
    const int chunk = blockIdx.x;
    const int inputMode = blockIdx.y;
    const int outputMode = blockIdx.z;
    const int thread = threadIdx.x;
    const size_t pixel = static_cast<size_t>(chunk) * blockSize + thread;

    double realSum = 0.0;
    double imagSum = 0.0;
    double normSum = 0.0;
    if (pixel < pixelCount) {
        const cuDoubleComplex field =
            fields[static_cast<size_t>(inputMode) * pixelCount + pixel];
        const cuDoubleComplex basis =
            modes[static_cast<size_t>(outputMode) * pixelCount + pixel];
        const cuDoubleComplex product = cuCmul(cuConj(basis), field);
        realSum = cuCreal(product);
        imagSum = cuCimag(product);
        normSum = cuCreal(field) * cuCreal(field) +
                  cuCimag(field) * cuCimag(field);
    }

    sums[thread] = realSum;
    sums[blockSize + thread] = imagSum;
    sums[2 * blockSize + thread] = normSum;
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            sums[thread] += sums[thread + stride];
            sums[blockSize + thread] += sums[blockSize + thread + stride];
            sums[2 * blockSize + thread] +=
                sums[2 * blockSize + thread + stride];
        }
        __syncthreads();
    }

    if (thread == 0) {
        const size_t pair =
            static_cast<size_t>(inputMode) * modeCount + outputMode;
        const size_t partialIndex = pair * chunkCount + chunk;
        partials[3 * partialIndex] = sums[0];
        partials[3 * partialIndex + 1] = sums[blockSize];
        partials[3 * partialIndex + 2] = sums[2 * blockSize];
    }
}

__global__ void storeProjections(const double* partials,
                                 cuDoubleComplex* result, int modeCount,
                                 int chunkCount, int frequencyIndex,
                                 int frequencyCount, int distanceIndex,
                                 int distanceCount) {
    const size_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t pairCount =
        static_cast<size_t>(modeCount) * modeCount;
    if (pair >= pairCount) {
        return;
    }

    double realSum = 0.0;
    double imagSum = 0.0;
    double normSum = 0.0;
    const size_t base = pair * chunkCount * 3;
    for (int chunk = 0; chunk < chunkCount; ++chunk) {
        const size_t index = base + static_cast<size_t>(chunk) * 3;
        realSum += partials[index];
        imagSum += partials[index + 1];
        normSum += partials[index + 2];
    }

    const double norm = sqrt(normSum);
    const cuDoubleComplex coefficient =
        make_cuDoubleComplex(realSum / norm, imagSum / norm);
    const int inputMode = pair / modeCount;
    const int outputMode = pair % modeCount;
    const size_t outputIndex =
        frequencyIndex +
        static_cast<size_t>(frequencyCount) *
            (distanceIndex +
             static_cast<size_t>(distanceCount) *
                 (outputMode + static_cast<size_t>(modeCount) * inputMode));
    result[outputIndex] = coefficient;
}

bool validDimensions(const data* d, int frequencyCount, int distanceCount) {
    if (d == nullptr || d->P == nullptr || d->modes == nullptr ||
        d->R == nullptr || d->kR == nullptr || d->Phi == nullptr ||
        d->diskmaskin == nullptr || d->ML <= 0 ||
        d->ML > std::numeric_limits<int>::max() || frequencyCount <= 0 ||
        distanceCount <= 0) {
        return false;
    }

    const int modeCount = static_cast<int>(d->ML);
    return d->R_rows > 0 && d->R_cols > 0 &&
           d->R_rows == d->n0 && d->R_cols == d->n1 &&
           d->R_rows == d->kR_rows && d->R_cols == d->kR_cols &&
           d->R_rows == d->Phi_rows && d->R_cols == d->Phi_cols &&
           d->R_rows == d->mask_rows && d->R_cols == d->mask_cols &&
           d->n0 == d->n1 && d->n2 == 1 && d->n3 == modeCount &&
           d->Pn0 == frequencyCount && d->Pn1 == distanceCount &&
           d->Pn2 == modeCount && d->Pn3 == modeCount;
}
}  // namespace

extern "C" int propagationCoeffs(const data* d, const double* frequencies,
                                  int frequencyCount, const double* distances,
                                  const double* tiltx, const double* tilty,
                                  int distanceCount, double eps) {
    if (!validDimensions(d, frequencyCount, distanceCount) ||
        frequencies == nullptr || distances == nullptr || tiltx == nullptr ||
        tilty == nullptr || !std::isfinite(eps) || eps < 0.0) {
        return -1;
    }

    const int modeCount = static_cast<int>(d->ML);
    const size_t pixelCount =
        static_cast<size_t>(d->R_rows) * d->R_cols;
    if (pixelCount > std::numeric_limits<size_t>::max() / modeCount) {
        return -1;
    }
    const size_t fieldCount = pixelCount * modeCount;
    const int chunkCount =
        static_cast<int>((pixelCount + blockSize - 1) / blockSize);
    const size_t partialCount =
        static_cast<size_t>(modeCount) * modeCount * chunkCount * 3;

    DeviceBuffer<cuDoubleComplex> dModes(fieldCount);
    DeviceBuffer<cuDoubleComplex> dFields(fieldCount);
    DeviceBuffer<double> dKR(pixelCount);
    DeviceBuffer<double> dR(pixelCount);
    DeviceBuffer<double> dPhi(pixelCount);
    DeviceBuffer<bool> dMask(pixelCount);
    DeviceBuffer<double> dPartials(partialCount);
    const size_t outputCount =
        static_cast<size_t>(frequencyCount) * distanceCount * modeCount *
        modeCount;
    DeviceBuffer<cuDoubleComplex> dResult(outputCount);
    if (dModes.status() != cudaSuccess || dFields.status() != cudaSuccess ||
        dKR.status() != cudaSuccess || dR.status() != cudaSuccess ||
        dPhi.status() != cudaSuccess || dMask.status() != cudaSuccess ||
        dPartials.status() != cudaSuccess ||
        dResult.status() != cudaSuccess) {
        return -2;
    }

    if (cudaMemcpy(dModes.get(), d->modes, fieldCount * sizeof(cuDoubleComplex),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dKR.get(), d->kR, pixelCount * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dR.get(), d->R, pixelCount * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dPhi.get(), d->Phi, pixelCount * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dMask.get(), d->diskmaskin, pixelCount * sizeof(bool),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return -2;
    }

    int dimensions[2] = {d->R_cols, d->R_rows};
    cufftHandle fftPlan;
    if (cufftPlanMany(&fftPlan, 2, dimensions, nullptr, 1, 0, nullptr, 1, 0,
                      CUFFT_Z2Z, modeCount) != CUFFT_SUCCESS) {
        return -3;
    }

    const dim3 fieldGrid(
        static_cast<unsigned int>((fieldCount + blockSize - 1) / blockSize));
    const dim3 reductionGrid(chunkCount, modeCount, modeCount);
    const dim3 projectionGrid(
        static_cast<unsigned int>((modeCount * modeCount + blockSize - 1) /
                                  blockSize));
    const double inversePixelCount = 1.0 / static_cast<double>(pixelCount);
    const double twoPi = 6.283185307179586476925286766559;

    int status = 0;
    for (int f = 0; f < frequencyCount && status == 0; ++f) {
        if (!std::isfinite(frequencies[f])) {
            status = -1;
            break;
        }
        const double k0 = twoPi * frequencies[f] / speedOfLight * sqrt(eps);
        for (int distance = 0; distance < distanceCount; ++distance) {
            if (!std::isfinite(distances[distance]) ||
                !std::isfinite(tiltx[distance]) ||
                !std::isfinite(tilty[distance])) {
                status = -1;
                break;
            }

            if (cudaMemcpy(dFields.get(), dModes.get(),
                           fieldCount * sizeof(cuDoubleComplex),
                           cudaMemcpyDeviceToDevice) != cudaSuccess) {
                status = -2;
                break;
            }
            if (cufftExecZ2Z(fftPlan, dFields.get(), dFields.get(),
                             CUFFT_FORWARD) != CUFFT_SUCCESS) {
                status = -4;
                break;
            }

            applySpectrum<<<fieldGrid, blockSize>>>(
                dFields.get(), dKR.get(), pixelCount, modeCount, k0,
                distances[distance]);
            if (cudaGetLastError() != cudaSuccess) {
                status = -5;
                break;
            }
            if (cufftExecZ2Z(fftPlan, dFields.get(), dFields.get(),
                             CUFFT_INVERSE) != CUFFT_SUCCESS) {
                status = -4;
                break;
            }

            applyTiltAndMask<<<fieldGrid, blockSize>>>(
                dFields.get(), dR.get(), dPhi.get(), dMask.get(), pixelCount,
                modeCount, k0, tiltx[distance], tilty[distance],
                inversePixelCount);
            if (cudaGetLastError() != cudaSuccess) {
                status = -5;
                break;
            }
            reduceProjections<<<reductionGrid, blockSize>>>(
                dFields.get(), dModes.get(), pixelCount, modeCount, chunkCount,
                dPartials.get());
            if (cudaGetLastError() != cudaSuccess) {
                status = -5;
                break;
            }
            storeProjections<<<projectionGrid, blockSize>>>(
                dPartials.get(), dResult.get(), modeCount, chunkCount, f,
                frequencyCount, distance, distanceCount);
            if (cudaGetLastError() != cudaSuccess) {
                status = -5;
                break;
            }
        }
    }

    if (status == 0 && cudaDeviceSynchronize() != cudaSuccess) {
        status = -5;
    }
    if (cufftDestroy(fftPlan) != CUFFT_SUCCESS && status == 0) {
        status = -4;
    }
    if (status != 0) {
        return status;
    }

    if (cudaMemcpy(d->P, dResult.get(),
                   outputCount * sizeof(cuDoubleComplex),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return -2;
    }
    return 0;
}
