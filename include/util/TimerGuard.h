#ifndef XI_TIMER_GUARD_H
#define XI_TIMER_GUARD_H

#include <fmt/chrono.h>


namespace xi
{

using FloatMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;

using FloatSeconds = std::chrono::duration<float, std::chrono::seconds::period>;


template <typename Duration = FloatMilliseconds, typename Clock = std::chrono::high_resolution_clock>
class TimerGuard
{
public:
    using TimePoint = typename Clock::time_point;

    /// @brief RAII-style timer guard.
    /// @param d      Duration (time elapsed from construction to destruction).
    /// @param stream Print time elapsed to stream.
    explicit TimerGuard(Duration * d = nullptr, std::FILE * stream = stdout)
            : d(d), startTime(Clock::now()), stream(stream)
    {
        // Do nothing.
    }

    ~TimerGuard()
    {
        auto timeElapsed = std::chrono::duration_cast<Duration>(Clock::now() - startTime);

        if (d)
        {
            *d = timeElapsed;
        }

        if (stream)
        {
            fmt::print(stream, "{}\n", timeElapsed);
        }
    }

private:
    Duration * d;
    TimePoint startTime;
    std::FILE * stream;
};


// // Use explicit instantiation only when needed.
// // One (and exactly one) manual instantiation will be needed for any instance.
// // Explicit instantiation declaration (extern template):
// // Skips implicit instantiation step.
// extern template class TimerGuard<FloatMilliseconds, std::chrono::high_resolution_clock>;
// // Explicit instantiation difinition:
// template class TimerGuard<FloatMilliseconds, std::chrono::high_resolution_clock>;

}  // namespace xi


#endif  // XI_TIMER_GUARD_H