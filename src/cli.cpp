#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include "inference.hpp"
#include "processing/image.hpp"
#include "package_paths.hpp"

namespace fs = std::filesystem;

namespace {

std::atomic<bool> stop_requested{false};

/**
 * @brief Convert SIGINT/SIGTERM into a cooperative Console shutdown request.
 * @param signal_number POSIX signal number; its value does not change behavior.
 */
void HandleSignal(int signal_number) {
  (void)signal_number;
  stop_requested = true;
}

/**
 * @brief Remove surrounding whitespace from an interactive Console command.
 * @param value Raw command text.
 * @return Trimmed command text.
 */
std::string Trim(std::string value) {
  const size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

/**
 * @brief Parse and validate the path argument following /image or /video.
 * @param line Complete interactive command.
 * @param command Matched command prefix.
 * @return Unquoted non-empty media path.
 */
std::string PathArgument(const std::string& line, const std::string& command) {
  std::string value = Trim(line.substr(command.size()));
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') ||
       (value.front() == '\'' && value.back() == '\''))) {
    value = value.substr(1, value.size() - 2);
  }
  if (value.empty()) throw std::invalid_argument(command + " requires a path");
  return value;
}

/**
 * @brief Check whether a line is one of the supported task commands.
 * @param value Trimmed interactive command.
 * @return True when a public LocateAnything task prefix matches.
 */
bool IsTaskCommand(const std::string& value) {
  static const std::string commands[] = {
      "/detect", "/ground", "/ground_single", "/gui", "/gui_box",
      "/text",   "/ground_text", "/layout", "/point"};
  return std::any_of(std::begin(commands), std::end(commands),
                     [&](const std::string& command) {
                       return value == command || value.rfind(command + " ", 0) == 0;
                     });
}

/**
 * @brief Match a Console command exactly or with a space-delimited argument.
 * @param value Trimmed interactive command.
 * @param command Command prefix to match.
 * @return True when value belongs to command.
 */
bool IsCommand(const std::string& value, const std::string& command) {
  return value == command || value.rfind(command + " ", 0) == 0;
}

struct Colors {
  std::string reset;
  std::string bold;
  std::string dim;
  std::string cyan;
  std::string green;
  std::string yellow;
  std::string blue;
  std::string magenta;
  std::string red;
};

/** Select ANSI colors only when output is an interactive terminal. */
Colors TerminalColors() {
  if (!isatty(STDOUT_FILENO) || std::getenv("NO_COLOR") != nullptr) return {};
  return {"\033[0m", "\033[1m", "\033[2m", "\033[36m", "\033[32m",
          "\033[33m", "\033[34m", "\033[35m", "\033[31m"};
}

/**
 * @brief Print the fixed LocateAnything Console banner.
 * @param color Terminal color palette, empty for plain output.
 */
void PrintBanner(const Colors& color) {
  static const char* lines[] = {
      "  ██╗      ██████╗  ██████╗ █████╗ ████████╗███████╗",
      "  ██║     ██╔═══██╗██╔════╝██╔══██╗╚══██╔══╝██╔════╝",
      "  ██║     ██║   ██║██║     ███████║   ██║   █████╗  ",
      "  ██║     ██║   ██║██║     ██╔══██║   ██║   ██╔══╝  ",
      "  ███████╗╚██████╔╝╚██████╗██║  ██║   ██║   ███████╗",
      "  ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝",
      "  █████╗ ███╗   ██╗██╗   ██╗████████╗██╗  ██╗██╗███╗   ██╗ ██████╗",
      " ██╔══██╗████╗  ██║╚██╗ ██╔╝╚══██╔══╝██║  ██║██║████╗  ██║██╔════╝",
      " ███████║██╔██╗ ██║ ╚████╔╝    ██║   ███████║██║██╔██╗ ██║██║  ███╗",
      " ██╔══██║██║╚██╗██║  ╚██╔╝     ██║   ██╔══██║██║██║╚██╗██║██║   ██║",
      " ██║  ██║██║ ╚████║   ██║      ██║   ██║  ██║██║██║ ╚████║╚██████╔╝",
      " ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝"};
  for (const char* line : lines) {
    std::cout << color.bold << color.cyan << line << color.reset << '\n';
  }
}

/**
 * @brief Print interactive task, media, and session commands.
 * @param color Terminal color palette, empty for plain output.
 */
void PrintHelp(const Colors& color) {
  std::cout << color.bold << color.cyan << "Tasks" << color.reset << '\n'
            << "  /detect cat,dog              目标检测\n"
            << "  /ground <query>[,<query>...] 指代表达，多查询\n"
            << "  /ground_single <query>[,...] 指代表达，单目标查询\n"
            << "  /gui <query>[,<query>...]    GUI 点定位\n"
            << "  /gui_box <query>[,<query>...] GUI 框定位\n"
            << "  /text                        文本 OCR\n"
            << "  /ground_text <query>[,...]   指定文本定位\n"
            << "  /layout title,table,figure   文档版面分析\n"
            << "  /point <query>[,<query>...]  通用点定位\n"
            << color.bold << color.cyan << "Session" << color.reset << '\n'
            << "  /image <image_path>          加载图片\n"
            << "  /video <video_path>          加载视频并处理全部帧\n"
            << "  regen                        重跑上次请求\n"
            << "  reset                        清除当前媒体\n"
            << "  exit                         退出程序\n";
}

struct ConsoleOptions {
  fs::path config;
  fs::path model_directory;
  fs::path tokenizer_directory;
  fs::path output_directory = "outputs";
  std::string vision_model = "LocateAnything-3B_vision_336x336.hbm";
  std::string language_model = "LocateAnything-3B_language_336x336.hbm";
  std::string embeddings = "LocateAnything-3B_embed_tokens.bin";
  int image_width = 336;
  int image_height = 336;
  std::string generation_mode = "hybrid";
  std::string l2m_sizes = "6:6:6:6";
  int max_new_tokens = 768;
  uint32_t vision_backend_mask = 15;
  uint32_t language_backend_mask = 15;
  float nms_iou = 0.9f;
};

/** Print the process-level command-line usage. */
void PrintUsage() {
  std::cout << "usage: console [--config FILE]\n";
}

/**
 * @brief Load shared runtime settings from the ROS-compatible YAML file.
 * @param path Explicit configuration file path.
 * @param options Destination options initialized with defaults by the caller.
 */
void LoadConfig(const fs::path& path, ConsoleOptions* options) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path.string());
  } catch (const YAML::Exception& error) {
    throw std::runtime_error("cannot read console config " + path.string() +
                             ": " + error.what());
  }
  YAML::Node parameters = root["hobot_locateanything"]["ros__parameters"];
  if (!parameters || !parameters.IsMap()) {
    throw std::runtime_error(
        "console config must contain hobot_locateanything.ros__parameters");
  }
  auto read_string = [&](const char* name, std::string* target) {
    if (parameters[name]) *target = parameters[name].as<std::string>();
  };
  auto read_path = [&](const char* name, fs::path* target, bool allow_empty) {
    if (!parameters[name]) return;
    const std::string value = parameters[name].as<std::string>();
    if (!value.empty() || allow_empty) *target = value;
  };
  read_path("model_directory", &options->model_directory, false);
  read_path("tokenizer_directory", &options->tokenizer_directory, true);
  read_path("output_directory", &options->output_directory, false);
  read_string("vision_model", &options->vision_model);
  read_string("language_model", &options->language_model);
  read_string("embeddings", &options->embeddings);
  read_string("generation_mode", &options->generation_mode);
  read_string("l2m_sizes", &options->l2m_sizes);
  if (parameters["max_new_tokens"]) {
    options->max_new_tokens = parameters["max_new_tokens"].as<int>();
  }
  if (parameters["image_width"]) {
    options->image_width = parameters["image_width"].as<int>();
  }
  if (parameters["image_height"]) {
    options->image_height = parameters["image_height"].as<int>();
  }
  if (parameters["vision_backend_mask"]) {
    options->vision_backend_mask =
        parameters["vision_backend_mask"].as<uint32_t>();
  }
  if (parameters["language_backend_mask"]) {
    options->language_backend_mask =
        parameters["language_backend_mask"].as<uint32_t>();
  }
  if (parameters["nms_iou"]) {
    options->nms_iou = parameters["nms_iou"].as<float>();
  }
}

/**
 * @brief Parse --config and validate the resulting Console settings.
 * @param argc Process argument count.
 * @param argv Process argument values.
 * @return Validated absolute-path Console settings.
 */
ConsoleOptions ParseArguments(int argc, char** argv) {
  ConsoleOptions options;
  options.config =
      locateanything::PackageRuntimeDirectory() / "config" / "config.yaml";
  options.model_directory = "models";
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      PrintUsage();
      std::exit(0);
    }
    if (argument == "--config" || argument == "-c") {
      if (++index >= argc) {
        throw std::invalid_argument(argument + " requires a value");
      }
      options.config = argv[index];
    }
  }
  options.config = fs::absolute(options.config);
  LoadConfig(options.config, &options);
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--config" || argument == "-c") {
      ++index;
      continue;
    }
    if (argument != "--help" && argument != "-h") {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  if (!(options.nms_iou > 0.0f && options.nms_iou <= 1.0f)) {
    throw std::invalid_argument("nms_iou must be in (0, 1]");
  }
  if (options.max_new_tokens <= 0) {
    throw std::invalid_argument("max_new_tokens in config must be positive");
  }
  locateanything::VisionProfile(options.image_width, options.image_height);
  if (options.generation_mode != "hybrid" && options.generation_mode != "slow") {
    throw std::invalid_argument(
        "generation_mode in config must be hybrid or slow");
  }
  options.model_directory =
      locateanything::ResolveRuntimePath(options.model_directory);
  if (options.tokenizer_directory.empty()) {
    throw std::invalid_argument(
        "tokenizer_directory must be set explicitly in config");
  }
  options.tokenizer_directory =
      locateanything::ResolveRuntimePath(options.tokenizer_directory);
  options.output_directory = fs::absolute(options.output_directory);
  return options;
}

enum class MediaKind { kNone, kImage, kVideo };

struct Media {
  MediaKind kind = MediaKind::kNone;
  fs::path path;
};

struct Request {
  Media media;
  std::string command;
};

/**
 * @brief Create the stable output directory for one media file.
 * @param options Console settings containing the output root.
 * @param source Input image or video path.
 * @return Created output directory path.
 */
fs::path OutputPath(const ConsoleOptions& options, const fs::path& source) {
  const std::string name = source.stem().string();
  if (name.empty() || name == "." || name == "..") {
    throw std::runtime_error("input file has no usable output name: " +
                             source.string());
  }
  const fs::path path = options.output_directory / name;
  fs::create_directories(path);
  return path;
}

/**
 * @brief Print compact performance and result summaries after one inference.
 * @param output Shared-core inference output.
 * @param color Terminal color palette, empty for plain output.
 */
void PrintPerformance(const locateanything::InferenceOutput& output,
                      const Colors& color) {
  const auto& metrics = output.metrics;
  const auto& language = metrics.language;
  const double tokens_per_second =
      language.decode_ms > 0.0
          ? static_cast<double>(language.generated_tokens) * 1000.0 / language.decode_ms
          : 0.0;
  std::set<std::string> labels;
  for (const auto& detection : output.prediction.detections) labels.insert(detection.label);
  for (const auto& point : output.prediction.points) labels.insert(point.label);
  std::ostringstream label_text;
  for (const std::string& label : labels) {
    if (label_text.tellp() > 0) label_text << ", ";
    label_text << label;
  }
  std::cout << color.bold << color.cyan << "Performance" << color.reset << '\n'
            << "  Vision   " << std::fixed << std::setprecision(1) << metrics.vision_ms
            << " ms\n"
            << "  Prefill  " << language.prefill_ms << " ms  "
            << language.prompt_tokens << " tokens\n"
            << "  Decode   " << language.decode_ms << " ms  "
            << language.generated_tokens << " tokens  " << tokens_per_second
            << " tokens/s\n"
            << "  Host     " << language.host_decode_ms << " ms\n"
            << "  Total    " << metrics.total_ms << " ms\n"
            << color.bold << color.cyan << "Result" << color.reset << '\n'
            << "  Labels " << color.magenta
            << (labels.empty() ? "(none)" : label_text.str()) << color.reset
            << "  |  Boxes " << color.green << output.prediction.detections.size()
            << color.reset << "  |  Points " << color.green
            << output.prediction.points.size() << color.reset << "  |  Stop "
            << output.stop_reason << '\n';
}

class Console {
 public:
  /**
   * @brief Create a Console around the shared inference core.
   * @param options Validated Console configuration.
   */
  explicit Console(ConsoleOptions options)
      : options_(std::move(options)),
        color_(TerminalColors()),
        session_(BuildInferenceOptions()) {}

  /** Run initialization and the interactive command loop until shutdown. */
  int Run() {
    PrintBanner(color_);
    const auto initialization_started = std::chrono::steady_clock::now();
    InitializeSession(initialization_started);
    PrintInitializationComplete(initialization_started);
    std::cout << color_.green << "Ready" << color_.reset
              << "  S600/Nash-P  |  " << options_.generation_mode
              << "  |  max tokens " << options_.max_new_tokens << '\n';
    PrintHelp(color_);

    std::string line;
    while (!stop_requested) {
      std::cout << color_.bold << color_.blue << "[User] <<< " << color_.reset
                << std::flush;
      if (!std::getline(std::cin, line)) break;
      line = Trim(line);
      if (line.empty()) continue;
      try {
        if (line == "exit" || line == "quit") break;
        if (line == "/help" || line == "help" || line == "?") {
          PrintHelp(color_);
        } else if (line == "reset") {
          media_ = {};
          last_request_ = {};
          has_last_request_ = false;
          std::cout << color_.green << "Media cleared" << color_.reset << '\n';
        } else if (line == "regen") {
          if (!has_last_request_) {
            std::cout << color_.yellow << "No previous request" << color_.reset << '\n';
          } else {
            Execute(last_request_.media, last_request_.command, false);
          }
        } else if (IsCommand(line, "/image")) {
          LoadImage(PathArgument(line, "/image"));
        } else if (IsCommand(line, "/video")) {
          LoadVideo(PathArgument(line, "/video"));
        } else if (IsTaskCommand(line)) {
          if (media_.kind == MediaKind::kNone) {
            std::cout << color_.yellow
                      << "Load media first with /image or /video" << color_.reset
                      << '\n';
          } else {
            Execute(media_, line, true);
          }
        } else {
          std::cout << color_.red << "Unsupported command" << color_.reset << '\n';
        }
      } catch (const std::exception& error) {
        std::cerr << color_.red << "Request failed: " << color_.reset << error.what()
                  << '\n';
      }
    }
    std::cout << color_.dim << "Session closed" << color_.reset << '\n';
    return 0;
  }

 private:
  /**
   * @brief Load both HBM files while a lightweight thread refreshes the UI.
   * @param started Monotonic initialization start time.
   */
  void InitializeSession(
      const std::chrono::steady_clock::time_point started) {
    std::mutex state_mutex;
    std::condition_variable state_changed;
    bool loading = true;
    std::string stage = "Starting";

    std::thread renderer([&] {
      std::unique_lock<std::mutex> lock(state_mutex);
      while (loading) {
        const std::string current_stage = stage;
        lock.unlock();
        PrintInitializationProgress(current_stage, started);
        lock.lock();
        state_changed.wait_for(lock, std::chrono::milliseconds(100), [&] {
          return !loading || stage != current_stage;
        });
      }
    });

    try {
      session_.Initialize([&](const std::string& current_stage) {
        {
          std::lock_guard<std::mutex> lock(state_mutex);
          stage = current_stage;
        }
        state_changed.notify_one();
      });
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(state_mutex);
        loading = false;
      }
      state_changed.notify_one();
      renderer.join();
      throw;
    }
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      loading = false;
    }
    state_changed.notify_one();
    renderer.join();
  }

  /**
   * @brief Render a moving initialization bar and elapsed seconds.
   * @param stage Current model-loading stage.
   * @param started Monotonic initialization start time.
   */
  void PrintInitializationProgress(
      const std::string& stage,
      const std::chrono::steady_clock::time_point started) {
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    if (color_.reset.empty()) {
      if (stage != loading_stage_) {
        loading_stage_ = stage;
        std::cout << "Loading " << stage << "..." << std::endl;
      }
      return;
    }
    constexpr int width = 28;
    constexpr int marker = 6;
    const int travel = width - marker;
    const int tick = static_cast<int>(elapsed * 10.0);
    int position = tick % (travel * 2);
    if (position > travel) position = travel * 2 - position;
    std::string bar(width, ' ');
    std::fill_n(bar.begin() + position, marker - 1, '=');
    bar[static_cast<size_t>(position + marker - 1)] = '>';
    std::cout << "\r\033[2K" << color_.yellow << "Loading " << std::left
              << std::setw(12) << stage << color_.reset << " [" << bar << "] "
              << std::right << std::fixed << std::setprecision(1) << elapsed
              << " s" << std::flush;
  }

  /**
   * @brief Replace the moving bar with the final HBM initialization status.
   * @param started Monotonic initialization start time.
   */
  void PrintInitializationComplete(
      const std::chrono::steady_clock::time_point started) const {
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    if (!color_.reset.empty()) std::cout << "\r\033[2K";
    std::cout << color_.green << "HBM loaded" << color_.reset
              << "  [============================] " << std::fixed
              << std::setprecision(1) << elapsed << " s\n";
  }

  /** Translate YAML Console settings into shared inference options. */
  locateanything::InferenceOptions BuildInferenceOptions() const {
    locateanything::InferenceOptions inference;
    if (setenv("HB_DNN_USER_DEFINED_L2M_SIZES", options_.l2m_sizes.c_str(), 1) != 0) {
      throw std::runtime_error("cannot configure S600 BPU L2 cache");
    }
    inference.vision_model =
        (options_.model_directory / options_.vision_model).string();
    inference.language_model =
        (options_.model_directory / options_.language_model).string();
    inference.embeddings =
        (options_.model_directory / options_.embeddings).string();
    inference.tokenizer_directory = options_.tokenizer_directory.string();
    inference.image_width = options_.image_width;
    inference.image_height = options_.image_height;
    inference.generation_mode = options_.generation_mode;
    inference.max_new_tokens = options_.max_new_tokens;
    inference.vision_backend_mask = options_.vision_backend_mask;
    inference.language_backend_mask = options_.language_backend_mask;
    inference.nms_iou = options_.nms_iou;
    return inference;
  }

  /**
   * @brief Resolve Console media without depending on the source directory.
   * @param value Absolute, working-directory-relative, or installed media path.
   * @return Normalized media path.
   */
  fs::path ResolveMediaPath(const std::string& value) const {
    const fs::path path(value);
    if (path.is_absolute()) return path.lexically_normal();
    if (fs::exists(path)) return fs::absolute(path).lexically_normal();
    return locateanything::ResolveRuntimePath(path);
  }

  /**
   * @brief Validate and remember one local image for the next task.
   * @param value User-provided image path.
   */
  void LoadImage(const std::string& value) {
    const fs::path path = ResolveMediaPath(value);
    if (!fs::is_regular_file(path) || cv::imread(path.string()).empty()) {
      throw std::runtime_error("image not found or unreadable: " + path.string());
    }
    media_ = {MediaKind::kImage, path};
    std::cout << color_.green << "Image loaded  " << color_.reset << path << '\n';
  }

  /**
   * @brief Validate and remember one local video for the next task.
   * @param value User-provided video path.
   */
  void LoadVideo(const std::string& value) {
    const fs::path path = ResolveMediaPath(value);
    cv::VideoCapture video(path.string());
    if (!fs::is_regular_file(path) || !video.isOpened()) {
      throw std::runtime_error("video not found or unreadable: " + path.string());
    }
    media_ = {MediaKind::kVideo, path};
    std::cout << color_.green << "Video loaded  " << color_.reset << path << '\n'
              << "  " << static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH)) << 'x'
              << static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT)) << "  |  "
              << std::fixed << std::setprecision(3) << video.get(cv::CAP_PROP_FPS)
              << " FPS  |  " << static_cast<int64_t>(video.get(cv::CAP_PROP_FRAME_COUNT))
              << " frames\n";
  }

  /**
   * @brief Dispatch a task to the selected media adapter.
   * @param media Selected image or video.
   * @param command Public LocateAnything task command.
   * @param remember Store this request for `regen` when true.
   */
  void Execute(const Media& media, const std::string& command, bool remember) {
    if (remember) {
      last_request_ = {media, command};
      has_last_request_ = true;
    }
    std::cout << color_.bold << color_.magenta << "[Assistant] >>> " << color_.reset
              << command << '\n';
    if (media.kind == MediaKind::kImage) {
      RunImage(media.path, command);
    } else {
      RunVideo(media.path, command);
    }
  }

  /**
   * @brief Run one local image and save annotated image and JSON result.
   * @param path Validated image path.
   * @param command Public LocateAnything task command.
   */
  void RunImage(const fs::path& path, const std::string& command) {
    const cv::Mat image = cv::imread(path.string());
    if (image.empty()) throw std::runtime_error("failed to read image: " + path.string());
    locateanything::InferenceOutputOptions output_options;
    output_options.render_annotated = true;
    output_options.serialize_json = true;
    output_options.pretty_json = true;
    locateanything::InferenceOutput output =
        session_.Infer(image, command, 1, output_options);
    const fs::path directory = OutputPath(options_, path);
    const fs::path annotated = directory / "annotated.jpg";
    const fs::path prediction = directory / "prediction.json";
    if (!cv::imwrite(annotated.string(), output.annotated_image)) {
      throw std::runtime_error("failed to save annotated image");
    }
    std::ofstream(prediction) << output.json << '\n';
    PrintPerformance(output, color_);
    std::cout << color_.bold << color_.cyan << "Saved" << color_.reset << '\n'
              << "  Image  " << annotated << '\n'
              << "  JSON   " << prediction << "\n\n";
  }

  /**
   * @brief Run every local-video frame and save media plus structured reports.
   * @param path Validated video path.
   * @param command Public LocateAnything task command applied to every frame.
   */
  void RunVideo(const fs::path& path, const std::string& command) {
    cv::VideoCapture video(path.string());
    if (!video.isOpened()) throw std::runtime_error("failed to open video: " + path.string());
    const int width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
    const int64_t expected = static_cast<int64_t>(video.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = video.get(cv::CAP_PROP_FPS);
    if (!(fps > 0.0)) fps = 25.0;

    const fs::path directory = OutputPath(options_, path);
    fs::path annotated = directory / "annotated.mp4";
    cv::VideoWriter writer(annotated.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(width, height));
    if (!writer.isOpened()) {
      annotated = directory / "annotated.avi";
      writer.open(annotated.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
                  cv::Size(width, height));
    }
    if (!writer.isOpened()) throw std::runtime_error("failed to create annotated video");
    const fs::path predictions = directory / "predictions.jsonl";
    std::ofstream jsonl(predictions);
    if (!jsonl) throw std::runtime_error("failed to create video prediction log");

    const auto started = std::chrono::steady_clock::now();
    uint64_t frame_index = 0;
    uint64_t boxes = 0;
    uint64_t points = 0;
    cv::Mat frame;
    while (!stop_requested && video.read(frame)) {
      locateanything::InferenceOutputOptions output_options;
      output_options.render_annotated = true;
      output_options.serialize_json = true;
      locateanything::InferenceOutput output =
          session_.Infer(frame, command, ++frame_index, output_options);
      writer.write(output.annotated_image);
      jsonl << output.json << '\n';
      boxes += output.prediction.detections.size();
      points += output.prediction.points.size();
      std::cout << '\r';
      if (!color_.reset.empty()) std::cout << "\033[2K";
      std::cout << color_.bold << color_.cyan << "Video" << color_.reset
                << "  frame " << color_.green << frame_index << color_.reset;
      if (expected > 0) std::cout << '/' << expected;
      std::cout << "  |  boxes " << boxes << "  |  " << std::fixed
                << std::setprecision(2) << output.metrics.total_ms / 1000.0
                << " s/frame" << std::flush;
    }
    std::cout << '\n';
    writer.release();
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    const fs::path summary = directory / "summary.json";
    std::ofstream(summary)
        << "{\"frames\":" << frame_index << ",\"boxes\":" << boxes
        << ",\"points\":" << points << ",\"elapsed_seconds\":" << std::fixed
        << std::setprecision(3) << elapsed << ",\"fps\":"
        << (elapsed > 0.0 ? frame_index / elapsed : 0.0) << "}\n";
    std::cout << color_.bold << color_.cyan << "Video complete" << color_.reset
              << "  Frames " << frame_index << "  |  Boxes " << boxes << "  |  "
              << std::fixed << std::setprecision(2) << elapsed << " s\n"
              << "  Video   " << annotated << '\n'
              << "  JSONL   " << predictions << '\n'
              << "  Summary " << summary << "\n\n";
  }

  ConsoleOptions options_;
  Colors color_;
  locateanything::InferenceSession session_;
  Media media_;
  Request last_request_;
  bool has_last_request_ = false;
  std::string loading_stage_;
};

}  // namespace

/**
 * @brief Run the independent Console process without initializing ROS.
 * @param argc Process argument count.
 * @param argv Process argument values.
 * @return Zero after normal shutdown, or one after a fatal error.
 */
int main(int argc, char** argv) {
  try {
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    ConsoleOptions options = ParseArguments(argc, argv);
    fs::create_directories(options.output_directory);
    return Console(std::move(options)).Run();
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}
