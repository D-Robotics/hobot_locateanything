#include "runner_process.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <utility>

#ifndef _WIN32
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace locateanything {
namespace {

std::vector<std::string> SplitTabs(const std::string& value) {
  std::vector<std::string> fields;
  size_t start = 0;
  while (true) {
    const size_t separator = value.find('\t', start);
    fields.push_back(value.substr(start, separator - start));
    if (separator == std::string::npos) return fields;
    start = separator + 1;
  }
}

}  // namespace

struct RunnerProcess::Impl {
#ifndef _WIN32
  pid_t pid = -1;
  FILE* input = nullptr;
  FILE* output = nullptr;
#endif
};

RunnerProcess::RunnerProcess() : impl_(std::make_unique<Impl>()) {}
RunnerProcess::~RunnerProcess() { Stop(); }
RunnerProcess::RunnerProcess(RunnerProcess&&) noexcept = default;
RunnerProcess& RunnerProcess::operator=(RunnerProcess&&) noexcept = default;

void RunnerProcess::Start(const std::string& program,
                          const std::vector<std::string>& arguments,
                          const std::string& ready_kind,
                          const std::function<void()>& wait_callback) {
#ifdef _WIN32
  (void)program;
  (void)arguments;
  (void)ready_kind;
  (void)wait_callback;
  throw std::runtime_error("HBM runner processes are supported on Linux only");
#else
  if (impl_->pid > 0) throw std::logic_error("runner process is already active");
  int parent_to_child[2];
  int child_to_parent[2];
  if (pipe(parent_to_child) != 0 || pipe(child_to_parent) != 0) {
    throw std::runtime_error("cannot create HBM runner pipes");
  }
  const pid_t pid = fork();
  if (pid < 0) throw std::runtime_error("cannot fork HBM runner");
  if (pid == 0) {
    dup2(parent_to_child[0], STDIN_FILENO);
    dup2(child_to_parent[1], STDOUT_FILENO);
    dup2(child_to_parent[1], STDERR_FILENO);
    close(parent_to_child[0]);
    close(parent_to_child[1]);
    close(child_to_parent[0]);
    close(child_to_parent[1]);
    std::vector<std::string> storage;
    storage.reserve(arguments.size() + 1);
    storage.push_back(program);
    storage.insert(storage.end(), arguments.begin(), arguments.end());
    std::vector<char*> argv;
    argv.reserve(storage.size() + 1);
    for (std::string& item : storage) argv.push_back(item.data());
    argv.push_back(nullptr);
    execvp(program.c_str(), argv.data());
    _exit(127);
  }

  close(parent_to_child[0]);
  close(child_to_parent[1]);
  impl_->pid = pid;
  impl_->input = fdopen(parent_to_child[1], "w");
  impl_->output = fdopen(child_to_parent[0], "r");
  if (impl_->input == nullptr || impl_->output == nullptr) {
    Stop();
    throw std::runtime_error("cannot open HBM runner streams");
  }
  setvbuf(impl_->input, nullptr, _IOLBF, 0);

  std::atomic<bool> waiting{true};
  std::thread progress;
  if (wait_callback) {
    progress = std::thread([&] {
      while (waiting.load(std::memory_order_relaxed)) {
        wait_callback();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    });
  }
  auto stop_progress = [&] {
    waiting.store(false, std::memory_order_relaxed);
    if (progress.joinable()) progress.join();
  };

  const std::string expected = "LAHBM/1\tREADY\t" + ready_kind;
  char* buffer = nullptr;
  size_t capacity = 0;
  while (true) {
    const ssize_t length = getline(&buffer, &capacity, impl_->output);
    if (length < 0) {
      free(buffer);
      stop_progress();
      Stop();
      throw std::runtime_error("HBM runner exited before it became ready");
    }
    std::string line(buffer, static_cast<size_t>(length));
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    if (line == expected) break;
  }
  free(buffer);
  stop_progress();
#endif
}

std::vector<std::string> RunnerProcess::Request(const std::string& request_id,
                                                const std::string& frame) {
#ifdef _WIN32
  (void)request_id;
  (void)frame;
  throw std::runtime_error("HBM runner processes are supported on Linux only");
#else
  if (impl_->pid <= 0 || impl_->input == nullptr || impl_->output == nullptr) {
    throw std::logic_error("HBM runner is not active");
  }
  if (frame.find('\n') != std::string::npos || frame.find('\r') != std::string::npos ||
      std::fprintf(impl_->input, "%s\n", frame.c_str()) < 0 ||
      std::fflush(impl_->input) != 0) {
    throw std::runtime_error("cannot send request to HBM runner");
  }

  char* buffer = nullptr;
  size_t capacity = 0;
  while (true) {
    const ssize_t length = getline(&buffer, &capacity, impl_->output);
    if (length < 0) {
      free(buffer);
      Stop();
      throw std::runtime_error("HBM runner exited during inference");
    }
    std::string line(buffer, static_cast<size_t>(length));
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    const std::vector<std::string> fields = SplitTabs(line);
    if (fields.size() >= 3 && fields[0] == "LAHBM/1" &&
        fields[2] == request_id) {
      if (fields[1] == "RESULT") {
        free(buffer);
        return fields;
      }
      if (fields[1] == "ERROR") {
        const std::string message = fields.size() >= 5 ? fields[4] : "HBM request failed";
        free(buffer);
        throw std::runtime_error(message);
      }
    }
  }
#endif
}

void RunnerProcess::Stop() {
#ifndef _WIN32
  if (impl_ == nullptr || impl_->pid <= 0) return;
  if (impl_->input != nullptr) {
    std::fprintf(impl_->input, "LAHBM/1\tQUIT\n");
    std::fflush(impl_->input);
    std::fclose(impl_->input);
    impl_->input = nullptr;
  }
  if (impl_->output != nullptr) {
    std::fclose(impl_->output);
    impl_->output = nullptr;
  }
  int status = 0;
  if (waitpid(impl_->pid, &status, WNOHANG) == 0) {
    kill(impl_->pid, SIGTERM);
    waitpid(impl_->pid, &status, 0);
  }
  impl_->pid = -1;
#endif
}

}  // namespace locateanything
