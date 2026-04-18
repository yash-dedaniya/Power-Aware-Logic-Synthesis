#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <cstdlib>
#include <random>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

// --- Configuration ---
const std::string VERILOG_DIR = "data";
const std::string OUTPUT_BENCH_DIR = "dataset/bench_files";
const std::string OUTPUT_LOG_DIR = "dataset/abc_logs";
const std::string TECH_LIB_PATH = "abc/NangateOpenCellLibrary_typical.lib";

const std::string ABC_BINARY = "abc/abc";

const int RECIPE_LENGTH = 20;
const int RECIPES_PER_CIRCUIT = 1000;
const int NUM_THREADS = std::thread::hardware_concurrency(); // Use all available CPU cores

const std::vector<std::string> COMMAND_POOL = {
    "rewrite", "rewrite -z", "refactor", "refactor -z", 
    "resub", "resub -z", "balance"
};

// --- Task Definition ---
struct Job {
    std::string verilog_path;
    std::string circuit_name;
    int run_id;
};

// --- Thread-Safe Message Queue ---
std::queue<Job> job_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;
bool production_complete = false;

// --- Helper: Generate Random Recipe ---
std::vector<std::string> generate_recipe() {
    thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(0, COMMAND_POOL.size() - 1);
    
    std::vector<std::string> recipe;
    for (int i = 0; i < RECIPE_LENGTH; ++i) {
        recipe.push_back(COMMAND_POOL[distribution(generator)]);
    }
    return recipe;
}

// --- Worker Thread Function ---
void worker_thread(int thread_id) {
    while (true) {
        Job current_job;
        
        // 1. Safely pull a job from the message queue
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [] { return !job_queue.empty() || production_complete; });
            
            if (production_complete && job_queue.empty()) {
                break; // No more jobs, exit thread
            }
            
            current_job = job_queue.front();
            job_queue.pop();
        }
        
        // 2. Build the exact interleaved ABC sequence
        std::vector<std::string> recipe = generate_recipe();
        std::string abc_script = "read " + current_job.verilog_path + "; strash; ";
        std::string recipe_log_str = "RECIPE: ";
        
        for (int step = 0; step < RECIPE_LENGTH; ++step) {
            std::string step_bench_path = OUTPUT_BENCH_DIR + "/" + 
                                          current_job.circuit_name + "_run" + 
                                          std::to_string(current_job.run_id) + "_step" + 
                                          std::to_string(step) + ".bench";
                                          
            abc_script += recipe[step] + "; write_bench " + step_bench_path + "; ";
            recipe_log_str += recipe[step] + "; ";
        }
        
        abc_script += "read_lib " + TECH_LIB_PATH + "; map; print_stats -p";
        
        // 3. Setup log file and execute using a shell redirect
        std::string log_path = OUTPUT_LOG_DIR + "/" + current_job.circuit_name + 
                               "_run" + std::to_string(current_job.run_id) + ".log";
                               
        // Write the recipe string to the log file first
        std::ofstream log_file(log_path);
        if (log_file.is_open()) {
            log_file << recipe_log_str << "\n";
            log_file.close();
        }

        // Redirect ABC's stdout and stderr to append to our log file
        std::string shell_command = ABC_BINARY + " -c \"" + abc_script + "\" >> " + log_path + " 2>&1";
        
        // Execute the system call
        int result = std::system(shell_command.c_str());
        
        if (result != 0) {
            std::cerr << "Thread " << thread_id << " failed on job: " << current_job.circuit_name 
                      << " run " << current_job.run_id << std::endl;
        }
    }
}

int main() {
    // Setup directories
    fs::create_directories(OUTPUT_BENCH_DIR);
    fs::create_directories(OUTPUT_LOG_DIR);

    // Read all .v files
    std::vector<fs::path> verilog_files;
    for (const auto& entry : fs::directory_iterator(VERILOG_DIR)) {
        if (entry.path().extension() == ".v") {
            verilog_files.push_back(entry.path());
        }
    }

    std::cout << "Found " << verilog_files.size() << " Verilog files.\n";
    std::cout << "Starting " << NUM_THREADS << " worker threads...\n";

    // Launch worker threads
    std::vector<std::thread> workers;
    for (int i = 0; i < NUM_THREADS; ++i) {
        workers.emplace_back(worker_thread, i);
    }

    // Produce tasks and push to the message queue
    int total_jobs = 0;
    int skipped_jobs = 0;
    
    for (const auto& v_file : verilog_files) {
        std::string circuit_name = v_file.stem().string();
        for (int run_id = 0; run_id < RECIPES_PER_CIRCUIT; ++run_id) {
            
            // --- NEW RESUME LOGIC ---
            std::string expected_log = OUTPUT_LOG_DIR + "/" + circuit_name + "_run" + std::to_string(run_id) + ".log";
            
            // If the log file already exists and isn't empty, skip this job
            if (fs::exists(expected_log) && fs::file_size(expected_log) > 0) {
                skipped_jobs++;
                continue; 
            }
            // ------------------------

            Job new_job{v_file.string(), circuit_name, run_id};
            
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                job_queue.push(new_job);
            }
            queue_cv.notify_one();
            total_jobs++;
        }
    }
    
    std::cout << "Skipped " << skipped_jobs << " already completed jobs.\n";

    // Signal workers that production is done
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        production_complete = true;
    }
    queue_cv.notify_all();

    std::cout << "Pushed " << total_jobs << " jobs to the queue. Waiting for completion...\n";

    // Wait for all threads to finish
    for (auto& worker : workers) {
        worker.join();
    }

    std::cout << "Data generation successfully completed.\n";
    return 0;
}