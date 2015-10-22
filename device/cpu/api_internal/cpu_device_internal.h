/*
Copyright (c) 2014, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "device/common/nn_device_internal.h"

#include <cstdint>

#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <memory>
#include <condition_variable>
#include <assert.h>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>
#endif

/* This file contains internal device structure implementation as well as thread pool class.

Thread pool usage:
    0. Thread pool object is accessible inside internal device implementation.
    1. Create 'job' vector that will contain all requests for thread pool.
    2. Each request should contain callback function that is able to 'unpack' opaque
       request_handle that is also sent inside request.
    3. Specific implementation of opaque internal structure can be made if required.
    4. After 'job' is created, use push_job function.
        - This function blocks execution until all worker threads complete their work.
        - You can't send 'job' that have more requests than there are threads available.
          Such action will result in assertion.
        - This function does not clear nor deallocate job vector, you must do it by yourself.
*/

// Internal implementation of request handle used by the thread pool.
struct nn_cpu_request_handle
{
    struct nn_workload_item *work_item;
    struct nn_workload_data_t *input_view;
    struct nn_workload_data_t *output_view;
};

// Generic request for thread pool.
struct nn_multithreaded_request
{
    // Callback that will be called with opaque request handle.
    std::function<void(void*)> callback;

    // Generic request handle.
    void* request_handle;
};

// Semaphore class used in thread pool implementation.
class nn_semaphore
{
public:
    // Setup semaphore and clear its counter.
    nn_semaphore()
        : mtx(),
          mtx_lock(mtx, std::defer_lock),
          count(0) {}

    // Set semaphore specific value that will be later decremented.
    void set_semaphore(size_t _count)
    {
        // Thread pool queue should be cleared which is indicated by semaphore value (==0).
        mtx_lock.lock();
        assert(count == 0);
        count = _count;
    }

    // Notifies waiters, decrements semaphore value and atomically clears working state
    // under semaphore lock. This method is called under wake lock also - as should
    // be every place changing request value.
    void atomic_clear_state_and_notify(nn_multithreaded_request** request)
    {
        std::unique_lock<std::mutex> lck(mtx);
        *request = nullptr;
        --count;
        cv.notify_one();
    }

    // Waits until tasks count will be equal to zero.
    void wait_for_all_and_clear()
    {
        cv.wait(mtx_lock, [this]() { return count == 0; });
        mtx_lock.unlock();
    }

    // Wait until any thread is done
    void wait_change()
    {
        cv.wait(mtx_lock);
    }

private:
    // This mutex is locked so 'count' member can't be accessed in more than one thread.
    std::mutex mtx;
    std::unique_lock<std::mutex> mtx_lock;
    std::condition_variable cv;

    // Number of tasks in pool queue.
    size_t count;
};

// Basic thread worker class.
class nn_thread_worker
{
public:
    nn_thread_worker(uint32_t id, nn_semaphore* semaphore)
        : hypervisor_semaphore(semaphore),
        current_request(nullptr),
        thread_id(id),
        worker_awaken(false),
        close_worker(false),
        worker_thread(&nn_thread_worker::task_loop, this)
    {
        // Wait until OS complete thread creation and call main thread function.
        while (!worker_awaken)
            ;
    }

    ~nn_thread_worker()
    {
        {
            // Set termination flag and wake up thread. Mutex will make
            // sure it wont happen during current task processing.
            std::lock_guard<std::mutex> wake_lock(wake_mutex);
            close_worker = true;
            wake_condition.notify_one();
        }

        // Wait for termination.
        worker_thread.join();
    }


    // Adds request to the thread.
    void add_request(nn_multithreaded_request* request)
    {
        if (request == nullptr) throw std::invalid_argument("null request sent");
        {
            // Locks wake mutex - it will get unlocked after worker
            // thread will cleanup itself after previous work.
            std::lock_guard<std::mutex> wake_lock(wake_mutex);
            current_request = request;
            wake_condition.notify_one();
        }
    }

    // Checks if there is any request processed.
    bool is_ready() const
    {
        return current_request == nullptr;
    }

#ifdef __linux__
    void get_affinity_np(size_t cpusetsize, cpu_set_t *cpuset)
    {
        int err =  pthread_getaffinity_np(worker_thread.native_handle(), sizeof(cpu_set_t), cpuset);
        if(err != 0)
        {
            throw std::runtime_error(std::string("Error getting affinity of thread. pthread_getaffinity_np error code: ") + std::to_string(err));
        }
    }

    void set_affinity_np(size_t cpusetsize, cpu_set_t *cpuset)
    {
        int err = pthread_setaffinity_np(worker_thread.native_handle(), sizeof(cpu_set_t), cpuset);
        if( err != 0 )
        {
            throw std::runtime_error(std::string("Error setting affinity of thread. pthread_setaffinity_np error code: ") + std::to_string(err));
        }
    }
#endif


private:
    // Main worker thread routine.
    void task_loop()
    {
        // Aquire wake mutex so no other thread can incorrectly interfere in thread values.
        std::unique_lock<std::mutex> wake_lock(wake_mutex);

        // Let constructor know that thread is fully created and safely locked.
        worker_awaken = true;

        // Main loop.
        while (!close_worker)
        {
            // Waits for notification that can be caused by new task or destructor.
            // When thread is in waiting state, it removes lock so other threads can push
            // new jobs into it or check its state. But after thread is notified, it locks
            // it again, so no code under this mutex can interfere during task processing.
            // It also causes thread to wait until all other threads changing its state
            // complete their work before thread starts.
            wake_condition.wait(wake_lock);

            // Safety check for spurious wake up or other wake up that shouldnt cause thread
            // to work on data - e.g. it could be termination call.
            bool assign_check;
            {
                assign_check = current_request != nullptr;

                if (assign_check)
                {
                    // Call user callback with sent request handle.
                    current_request->callback(current_request->request_handle);

                    // Safely clears request state and notifies pool semaphore.
                    hypervisor_semaphore->atomic_clear_state_and_notify(&current_request);
                }
            }
        }
    }

     // Semaphore of thread pool, shader by all threads.
    nn_semaphore* hypervisor_semaphore;

    // Value used both as request handle and indicator of thread state.
    nn_multithreaded_request* current_request;

    // ID of thread visible by thread pool.
    uint32_t thread_id;

    // Special controlling values.
    volatile bool worker_awaken;
    volatile bool close_worker;

    // Mutexes used by a thread.
    mutable std::mutex wake_mutex;
    std::condition_variable wake_condition;

    // Main object of worker thread.
    std::thread worker_thread;
};

// Class that provides Hardware specific info eg. number of logical cores and physical cores
// TODO: make it a singleton, Windows support
struct platform_info
{
    long num_logical_processors;
    long num_physical_processors_per_socket;
    long num_hw_threads_per_socket;
    unsigned int num_ht_threads;
    unsigned int num_total_phys_cores;
};


class nn_hardware_platform
{
    public:
        nn_hardware_platform() : m_num_logical_processors(0), m_num_physical_processors_per_socket(0), m_num_hw_threads_per_socket(0) ,m_num_ht_threads(1), m_num_total_phys_cores(1)
        {
#ifdef __linux__
            m_num_logical_processors = sysconf(_SC_NPROCESSORS_ONLN);

            m_num_physical_processors_per_socket = 0;

            std::ifstream ifs;
            ifs.open("/proc/cpuinfo");

            // If there is no /proc/cpuinfo fallback to default scheduler
            if(ifs.good() == false) {
                m_num_physical_processors_per_socket = m_num_logical_processors;
                assert(0);  // No cpuinfo? investigate that
                return;
            }
            std::string cpuinfo_content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            std::stringstream cpuinfo_stream(cpuinfo_content);
            std::string cpuinfo_line;
            while(std::getline(cpuinfo_stream,cpuinfo_line,'\n')){
                if(cpuinfo_line.find("cpu cores") != std::string::npos) {
                    // convert std::string into number eg. skip colon and after it in the same line  should be number of physical cores per socket
                    std::stringstream( cpuinfo_line.substr(cpuinfo_line.find(":") + 1) ) >> m_num_physical_processors_per_socket;
                    break;
                }
                if(cpuinfo_line.find("siblings") != std::string::npos) {
                    // convert std::string into number eg. skip colon and after it in the same line  should be number of HW threads per socket
                    std::stringstream( cpuinfo_line.substr(cpuinfo_line.find(":") + 1) ) >> m_num_hw_threads_per_socket;
                }
            }
            // There is cpuinfo, but parsing did not get quite right? Investigate it
            assert( m_num_physical_processors_per_socket > 0);
            assert( m_num_hw_threads_per_socket > 0);

            // Calculate how many threads can be run on single cpu core , in case of lack of hw info attributes assume 1
            m_num_ht_threads =  m_num_physical_processors_per_socket != 0 ? m_num_hw_threads_per_socket/ m_num_physical_processors_per_socket : 1;
            // calculate total number of physical cores eg. how many full Hw threads we can run in parallel
            m_num_total_phys_cores = m_num_hw_threads_per_socket != 0 ? m_num_logical_processors / m_num_hw_threads_per_socket * m_num_physical_processors_per_socket : 1;

            ifs.close();
#endif
        }
    void get_platform_info(platform_info& pi)
    {
       pi.num_logical_processors = m_num_logical_processors;
       pi.num_physical_processors_per_socket = m_num_physical_processors_per_socket;
       pi.num_hw_threads_per_socket = m_num_hw_threads_per_socket;
       pi.num_ht_threads = m_num_ht_threads;
       pi.num_total_phys_cores = m_num_total_phys_cores;
    }
    private:
        long m_num_logical_processors;
        long m_num_physical_processors_per_socket;
        long m_num_hw_threads_per_socket;
        unsigned int m_num_ht_threads;
        unsigned int m_num_total_phys_cores;
};

// Thread pool implementation.
class nn_thread_worker_pool
{
public:
    // Basic constructor.
    nn_thread_worker_pool(size_t cfg_num_threads = 0) : m_max_physical_threads(0)
    {
        size_t num_threads;

        if (cfg_num_threads == 0)
        {
            // Check system to get number of HW threads available.
            // TODO: add specific implementation for windows/linux to get exact value of cores.
            num_threads = std::thread::hardware_concurrency() / 2; // take half, to work on 1 of 2 sockets

        }
        else
        {
            // Get number of threads specified by user.
            num_threads = cfg_num_threads;
        }


        // If there is only one thread available - do not create
        // subthreads, pool will process all jobs on its own.
        if (num_threads > 1)
        {
            for (auto thread_id = 0; thread_id < num_threads; ++thread_id)
            {
                auto thread = std::unique_ptr<nn_thread_worker>(new nn_thread_worker(thread_id, &semaphore));
                threads.push_back(std::move(thread));
            }

#ifdef __linux__
        nn_hardware_platform hw_platform;
        platform_info hw_info;
        hw_platform.get_platform_info(hw_info);

        // Get original affinity of threads (assuming it is same for all threads)
        threads[0]->get_affinity_np(sizeof(cpu_set_t), &m_original_cpuset);

        // Create affinity mask by masking out bits responsible for logical threads to
        // avoid running more than one thread on the same physical core.
        // In other words affinity mask needs to modified so no more bits than one are set for
        // same physical core
        // This is useful for Fully connected layer under linux where linux scheduler
        // happens to run two threads on one core while having physical cores available and idle
        // Also count number of physical threads we can have to compare it at runtime
        // with number of working threads to be processing to decide if affinity should be changed for threads
        m_physical_cpuset = m_original_cpuset;
        if( hw_info.num_ht_threads > 1) {
            for( unsigned int i = 0; i < hw_info.num_logical_processors; ++i ) {
                if( CPU_ISSET(i,&m_physical_cpuset) ) {
                    ++m_max_physical_threads;
                    for(unsigned int j = i + hw_info.num_total_phys_cores; j < hw_info.num_logical_processors; ++j ) {
                        CPU_CLR(j,&m_physical_cpuset);
                    }
                }
            }
        }

        // verification (printing of set of physical mask) .
        // Please do nto remove it as it is helpful during development process!
        //printf("Affinity[%d]: ",0);
        //for(unsigned int j =0; j< hw_info.num_logical_processors; ++j) {
           //printf("%d,",CPU_ISSET(j,&m_physical_cpuset) == true ? 1 : 0);
        //}
        //printf("\n");
#else
    //TODO: windows
#endif
        }


    }

    ~nn_thread_worker_pool()
    {
    }

    // Get number of worker threads available.
    size_t get_num_threads()
    {
        // If there are no worker threads, then one thread is available - the pool thread.
        if (threads.size()) return threads.size();
        else return 1;
    }

    void push_job_on_physical_cores_if_beneficial(std::vector<nn_multithreaded_request>& requests)
    {
        // We have observed that linux task scheduler is distributing
        // computational tasks on the same physical CPU core when no
        // other important job is running on diffrent CPU core
        // which makes all computation running slower than expected
        // So in case of having enough CPU physical cores to run
        // threads separatly eg. one physical thread on one physical CPU core,
        // we modify affinity not allow scheduler to have to threads on the same physical CPU core
#ifdef __linux__
        // check if it is affordable to run job with set affinity
        if(m_max_physical_threads >= requests.size()) {
            this->push_job_on_physical_cores(requests);
        } else {
#endif
            this->push_job(requests);
#ifdef __linux
        }
#endif
    }


    // Push job queue setting calculated previously affinity
    void push_job_on_physical_cores(std::vector<nn_multithreaded_request>& requests)
    {
        // Sent requests to worker threads.
        if (threads.size() != 0)
        {
            // Setup semaphore and lock its mutex.
            semaphore.set_semaphore(requests.size());

            auto threads_begin = std::begin(threads);
            auto threads_end = std::end(threads);

            // Run tasks.
            for (auto& request : requests)
            {
                auto ready_thread = threads_end;

                // Find waiting thread.
                do{
                    ready_thread = std::find_if(threads_begin, threads_end,
                        [&](std::unique_ptr<nn_thread_worker> const& thread) { return thread->is_ready(); });
                    if (ready_thread != threads_end)
                        break;
                    semaphore.wait_change();
                } while (ready_thread == threads_end);

                // It will hit if there are no ready threads.
                assert(ready_thread != threads_end);

#ifdef __linux__
                // Set affintiy of thread so only one thread per core (physical cores) are allowed
                (**ready_thread).set_affinity_np(sizeof(cpu_set_t), &m_physical_cpuset);
#else
                // TODO: Windows
#endif
                // Add request to ready thread found.
                (**ready_thread).add_request(&request);
            }

            // Wait for all threads and clear semaphore locks.
            semaphore.wait_for_all_and_clear();

#ifdef __linux__
            // Revert original affinity mask
            for( auto& thread : threads)
            {
                (*thread).set_affinity_np(sizeof(cpu_set_t), &m_original_cpuset);
            }
#else
            // TODO: Windows
#endif

        }
        else
        {
            // Singlethreaded pool... run tasks sequentially by itself.
            for (auto& request : requests)
            {
                request.callback(request.request_handle);
            }
        }
    }

    // Push job queue.
    void push_job(std::vector<nn_multithreaded_request>& requests)
    {
        // Sent requests to worker threads.
        if (threads.size() != 0)
        {
            // Setup semaphore and lock its mutex.
            semaphore.set_semaphore(requests.size());

            auto threads_begin = std::begin(threads);
            auto threads_end = std::end(threads);

            // Run tasks.
            for (auto& request : requests)
            {
                auto ready_thread = threads_end;

                // Find waiting thread.
                do{
                    ready_thread = std::find_if(threads_begin, threads_end,
                        [&](std::unique_ptr<nn_thread_worker> const& thread) { return thread->is_ready(); });
                    if (ready_thread != threads_end)
                        break;
                    semaphore.wait_change();
                } while (ready_thread == threads_end);

                // It will hit if there are no ready threads.
                assert(ready_thread != threads_end);

                // Add request to ready thread found.
                (**ready_thread).add_request(&request);
            }

            // Wait for all threads and clear semaphore locks.
            semaphore.wait_for_all_and_clear();
        }
        else
        {
            // Singlethreaded pool... run tasks sequentially by itself.
            for (auto& request : requests)
            {
                request.callback(request.request_handle);
            }
        }
    }

private:

    // Main semaphore, visible by all worker threads.
    nn_semaphore semaphore;

    // Vector of worker threads.
    std::vector<std::unique_ptr<nn_thread_worker>> threads;

#ifdef __linux__
    cpu_set_t m_original_cpuset;
    cpu_set_t m_physical_cpuset;
#endif
    unsigned int m_max_physical_threads;    // Number of physical threads we can run on separate cpu cores. This value consider affinity mask set externaly
};

// Internal implementation of device structure.
struct nn_device_internal : nn_device_t
{
    nn_device_internal() : thread_pool() {};
    nn_device_internal(size_t num_threads) : thread_pool(num_threads) {};

    nn_thread_worker_pool thread_pool;
};

void copy_data(nn_device_internal *device, nn_data_t *destination, const nn_workload_data_t *source);
void copy_data(nn_device_internal *device, nn_workload_data_t *destination, const nn_data_t *source);
