#pragma once

#include <memory>
#include <string>
#include <vector>

namespace locateanything {

class RunnerProcess {
 public:
  RunnerProcess();
  ~RunnerProcess();
  RunnerProcess(RunnerProcess&&) noexcept;
  RunnerProcess& operator=(RunnerProcess&&) noexcept;
  RunnerProcess(const RunnerProcess&) = delete;
  RunnerProcess& operator=(const RunnerProcess&) = delete;

  void Start(const std::string& program,
             const std::vector<std::string>& arguments,
             const std::string& ready_kind);
  std::vector<std::string> Request(const std::string& request_id,
                                   const std::string& frame);
  void Stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace locateanything
