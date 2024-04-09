#pragma once
#define HIP_CALL(x)                                                            \
  {                                                                            \
    hipError_t err = x;                                                        \
    if (err != hipSuccess) {                                                   \
      std::cerr << "Error: " << hipGetErrorString(err) << " at " << __FILE__   \
                << ":" << __LINE__ << std::endl;                               \
      exit(err);                                                               \
    }                                                                          \
  }