#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <syncstream>
#include <thread>

enum class LogLevel : uint8_t {
  TRACE,
  DEBUG,
  INFO,
  WARN,
  ERROR,
};

inline std::string_view level_to_str(LogLevel lvl) {
  switch (lvl) {
    case LogLevel::TRACE:
      return "TRACE";
    case LogLevel::DEBUG:
      return "DEBUG";
    case LogLevel::INFO:
      return "INFO";
    case LogLevel::WARN:
      return "WARN";
    case LogLevel::ERROR:
      return "ERROR";
  }
  return "UNK";
}

inline const char* level_to_color(LogLevel lvl) {
  switch (lvl) {
    case LogLevel::TRACE:
      return "\033[2m"; // dim
    case LogLevel::DEBUG:
      return "\033[36m"; // cyan
    case LogLevel::INFO:
      return "\033[32m"; // green
    case LogLevel::WARN:
      return "\033[33m"; // yellow
    case LogLevel::ERROR:
      return "\033[31m"; // red
  }
  return "";
}

class Log {
public:
  explicit Log(LogLevel lvl = LogLevel::INFO) noexcept : lvl(lvl) {
    // us timestamp
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto us = duration_cast<microseconds>(now.time_since_epoch()).count();

    buffer << level_to_color(lvl) << "[" << std::setw(7) << std::setfill(' ') << level_to_str(lvl)
           << "] " << us << "us "
           << "(tid=" << std::this_thread::get_id() << ") "
           << "\033[0m"; // reset color
  }

  ~Log() noexcept {
    std::osyncstream out(std::cout);
    out << buffer.str() << '\n';
  }

  template <typename T> Log& operator<<(const T& v) {
    buffer << v;
    return *this;
  }

private:
  std::ostringstream buffer;
  LogLevel lvl;
};
