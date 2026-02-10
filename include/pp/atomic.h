#ifndef ATOMIC_H
#define ATOMIC_H

#include "defs.h"
#include "OpenMPBackend.h"
#include "KokkosBackend.h"
#include "hipBackend.h"
#include "cudaBackend.h"

namespace portableWrapper{
    namespace atomic {
      namespace accelerated {
        /**
         * Atomic addition operation
         * @param target The target variable to add to
         * @param value The value to add
         */
          template<typename T1, typename T2>
          DEVICEPREFIX void Add(T1& target, const T2& value){
            #if defined(USE_KOKKOS)
              kokkos::atomic::Add(target, value);
            #elif defined(USE_CUDA)
              cuda::atomic::Add(target, value);
            #elif defined(USE_HIP)
              hip::atomic::Add(target, value);
            #else
              openmp::atomic::Add(target, value);
            #endif
          }

          /**
           * Atomic AND operation
           * @param target The target variable to and with
           * @param value The value to and with
           */
          template<typename T1, typename T2>
          DEVICEPREFIX void And(T1& target, const T2& value){
            #if defined(USE_KOKKOS)
              kokkos::atomic::And(target, value);
            #elif defined(USE_CUDA)
              cuda::atomic::And(target, value);
            #elif defined(USE_HIP)
              hip::atomic::And(target, value);
            #else
              openmp::atomic::And(target, value);
            #endif
        }

        /**
         * Atomic decrement operation
         * @param target The target variable to decrement
         */
          template<typename T>
          DEVICEPREFIX void Dec(T& target){
            #if defined(USE_KOKKOS)
              kokkos::atomic::Dec(target);
            #elif defined(USE_CUDA)
              cuda::atomic::Dec(target);
            #elif defined(USE_HIP)
              hip::atomic::Dec(target);
            #else
              openmp::atomic::Dec(target);
            #endif
          }

          /**
           * Atomic increment operation
           */
          template<typename T>
          DEVICEPREFIX void Inc(T& target){
            #if defined(USE_KOKKOS)
              kokkos::atomic::Inc(target);
            #elif defined(USE_CUDA)
              cuda::atomic::Inc(target);
            #elif defined(USE_HIP)
              hip::atomic::Inc(target);
            #else
              openmp::atomic::Inc(target);
            #endif
          }

          /**
           * Atomic Max operation
           * @param target The target variable to perform the operation on
           * @param value The value to compare with
           */
          template<typename T, typename T2>
          DEVICEPREFIX void Max(T& target, const T2& value){
            #if defined(USE_KOKKOS)
              kokkos::atomic::Max(target, value);
            #elif defined(USE_CUDA)
              cuda::atomic::Max(target, value);
            #elif defined(USE_HIP)
              hip::atomic::Max(target, value);
            #else
              openmp::atomic::Max(target, value);
            #endif
          }

          /**
           * Atomic Min operation
           * @param target The target variable to perform the operation on
           * @param value The value to compare with
           */
          template<typename T, typename T2>
          DEVICEPREFIX void Min(T& target, const T2& value){
            #if defined(USE_KOKKOS)
              kokkos::atomic::Min(target, value);
            #elif defined(USE_CUDA)
              cuda::atomic::Min(target, value);
            #elif defined(USE_HIP)
              hip::atomic::Min(target, value);
            #else
              openmp::atomic::Min(target, value);
            #endif
          }

          /**
           * Atomic OR operation
           * @param target The target variable to perform the operation on
           * @param value The value to or with
           */
          template<typename T, typename T2>
          DEVICEPREFIX void Or(T& target, const T2& value){
            #if defined(USE_KOKKOS)
              kokkos::atomic::Or(target, value);
            #elif defined(USE_CUDA)
              cuda::atomic::Or(target, value);
            #elif defined(USE_HIP)
              hip::atomic::Or(target, value);
            #else
              openmp::atomic::Or(target, value);
            #endif
          }

          /**
           * Atomic subtraction operation
           * @param target The target variable to perform the operation on
           * @param value The value to subtract
           */
          template<typename T, typename T2>
          DEVICEPREFIX void Sub(T& target, const T2& value){
            #if defined(USE_KOKKOS)
              kokkos::atomic::Sub(target, value);
            #elif defined(USE_CUDA)
              cuda::atomic::Sub(target, value);
            #elif defined(USE_HIP)
              hip::atomic::Sub(target, value);
            #else
              openmp::atomic::Sub(target, value);
            #endif
          }
        } // namespace accelerated

        namespace host{
          /**
           * Atomic addition operation for host
           * @param target The target variable to add to
           * @param value The value to add
           */
          template<typename T1, typename T2>
          DEVICEPREFIX void Add(T1& target, const T2& value){
            //Host is always OpenMP
              openmp::atomic::Add(target, value);
          }

          /**
           * Atomic AND operation for host
           * @param target The target variable to and with
           * @param value The value to and with
           */
          template<typename T1, typename T2>
          DEVICEPREFIX void And(T1& target, const T2& value){
            //Host is always OpenMP
              openmp::atomic::And(target, value);
          }

          /**
           * Atomic decrement operation for host
           * @param target The target variable to decrement
           */
          template<typename T>
          DEVICEPREFIX void Dec(T& target){
            //Host is always OpenMP
              openmp::atomic::Dec(target);
          }

          /**
           * Atomic increment operation for host
           */
          template<typename T>
          DEVICEPREFIX void Inc(T& target){
            //Host is always OpenMP
              openmp::atomic::Inc(target);
          }

          /**
           * Atomic Max operation for host
           * @param target The target variable to perform the operation on
           * @param value The value to compare with
           */
          template<typename T, typename T2>
          DEVICEPREFIX void Max(T& target, const T2& value){
            //Host is always OpenMP
              openmp::atomic::Max(target, value);
          }

          /**
           * Atomic Min operation for host
           * @param target The target variable to perform the operation on
           * @param value The value to compare with
           */
          template<typename T, typename T2>
          DEVICEPREFIX void Min(T& target, const T2& value){
            //Host is always OpenMP
              openmp::atomic::Min(target, value);
          }

          /**
           * Atomic OR operation for host
           * @param target The target variable to perform the operation on
           * @param value The value to or with
           */
          template<typename T, typename T2>
          DEVICEPREFIX void Or(T& target, const T2& value){
            //Host is always OpenMP
              openmp::atomic::Or(target, value);
          }

          /**
           * Atomic subtraction operation for host
           * @param target The target variable to perform the operation on
           * @param value The value to subtract
           */
          template<typename T, typename T2>
          DEVICEPREFIX void Sub(T& target, const T2& value){
            //Host is always OpenMP
              openmp::atomic::Sub(target, value);
          }
        } // namespace host
      } // namespace atomic
    } // namespace portableWrapper
#endif