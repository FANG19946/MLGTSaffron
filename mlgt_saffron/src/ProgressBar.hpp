#ifndef PROGRESS_BAR_HPP
#define PROGRESS_BAR_HPP

#include <iostream>
#include <iomanip>
#include <mutex>
#include <cstdint>

class ProgressBar {
private:
    uint64_t total_;
    uint64_t last_percent_;
    std::mutex mutex_;

public:
    explicit ProgressBar(uint64_t total)
        : total_(total), last_percent_(0) {}

    void update(uint64_t current)
    {
        if (total_ == 0) return;

        uint64_t percent = (current * 100) / total_;

        if (percent == last_percent_ && current != total_)
            return;

        std::lock_guard<std::mutex> lock(mutex_);

        last_percent_ = percent;

        constexpr int BAR_WIDTH = 50;

        int filled = static_cast<int>((percent * BAR_WIDTH) / 100);

        std::cout << "\r[";

        for (int i = 0; i < BAR_WIDTH; i++)
            std::cout << (i < filled ? '=' : ' ');

        std::cout << "] "
                  << std::setw(3) << percent << "% ("
                  << current << "/" << total_ << ")"
                  << std::flush;

        if (current == total_)
            std::cout << std::endl;
    }
};

#endif