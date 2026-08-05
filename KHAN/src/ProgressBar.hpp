#ifndef PROGRESS_BAR_HPP
#define PROGRESS_BAR_HPP
#include <chrono>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <cstdint>

class ProgressBar {
private:
    uint64_t total_;
    uint64_t last_percent_;
    std::mutex mutex_;
    std::chrono::steady_clock::time_point start_;

public:
    explicit ProgressBar(uint64_t total)
        : total_(total), last_percent_(0), start_(std::chrono::steady_clock::now()) {}

    void update(uint64_t current)
    {
        if (total_ == 0) return;

        uint64_t percent = (current * 100) / total_;

        std::lock_guard<std::mutex> lock(mutex_);

        if (percent == last_percent_ && current != total_)
            return;

        last_percent_ = percent;

        constexpr int BAR_WIDTH = 50;

        int filled = static_cast<int>((percent * BAR_WIDTH) / 100);

        auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_
        ).count();

        std::cout << "\r[";

        for (int i = 0; i < BAR_WIDTH; i++)
            std::cout << (i < filled ? '=' : ' ');

        std::cout << "] "
                  << std::setw(3) << percent << "% ("
                  << current << "/" << total_ << ") "
                  << "Elapsed: " << std::setw(6) << elapsed << "s"
                  << std::flush;

        if (current == total_)
            std::cout << std::endl;
    }
};

#endif