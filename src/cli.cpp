#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include "inference.hpp"

namespace fs = std::filesystem;

namespace {

std::atomic<bool> stop_requested{false};

void HandleSignal(int) { stop_requested = true; }

std::string Trim(std::string value) {
  const size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

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

bool IsTaskCommand(const std::string& value) {
  static const std::string commands[] = {
      "/detect", "/ground", "/ground_single", "/gui", "/gui_box",
      "/text",   "/ground_text", "/layout", "/point"};
  return std::any_of(std::begin(commands), std::end(commands),
                     [&](const std::string& command) {
                       return value == command || value.rfind(command + " ", 0) == 0;
                     });
}

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

Colors TerminalColors() {
  if (!isatty(STDOUT_FILENO) || std::getenv("NO_COLOR") != nullptr) return {};
  return {"\033[0m", "\033[1m", "\033[2m", "\033[36m", "\033[32m",
          "\033[33m", "\033[34m", "\033[35m", "\033[31m"};
}

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

void PrintHelp(const Colors& color) {
  std::cout << color.bold << color.cyan << "Tasks" << color.reset << '\n'
            << "  /detect cat,dog              目标检测\n"
            << "  /ground <phrase>             指代表达，多目标\n"
            << "  /ground_single <phrase>      指代表达，单目标\n"
            << "  /gui <element>               GUI 点定位\n"
            << "  /gui_box <element>           GUI 框定位\n"
            << "  /text                        文本 OCR\n"
            << "  /ground_text <text>          指定文本定位\n"
            << "  /layout title,table,figure   文档版面分析\n"
            << "  /point <target>              通用点定位\n"
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
  std::string vision_model = "LocateAnything-3B_vision.hbm";
  std::string language_model = "LocateAnything-3B_language.hbm";
  std::string embeddings = "LocateAnything-3B_embed_tokens.bin";
  std::string generation_mode = "hybrid";
  std::string l2m_sizes = "6:6:6:6";
  int max_new_tokens = 4096;
  uint32_t vision_backend_mask = 15;
  uint32_t language_backend_mask = 15;
  float nms_iou = 0.9f;
};

void PrintUsage() {
  std::cout << "usage: console [--config FILE]\n";
}

void LoadConfig(const fs::path& path, ConsoleOptions* options) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path.string());
  } catch (const YAML::Exception& error) {
    throw std::runtime_error("cannot read console config " + path.string() +
                             ": " + error.what());
  }
  YAML::Node parameters = root["hobot_locateanything"]["ros__parameters"];
  if (!parameters || !parameters.IsMap()) parameters = root["inference"];
  if (!parameters || !parameters.IsMap()) {
    throw std::runtime_error("console config must contain an inference map");
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

ConsoleOptions ParseArguments(int argc, char** argv) {
  ConsoleOptions options;
  options.config = "config.yaml";
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
  if (options.generation_mode != "hybrid" && options.generation_mode != "slow") {
    throw std::invalid_argument(
        "generation_mode in config must be hybrid or slow");
  }
  options.model_directory = fs::absolute(options.model_directory);
  if (!options.tokenizer_directory.empty()) {
    options.tokenizer_directory = fs::absolute(options.tokenizer_directory);
  }
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

std::string Timestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm local{};
  localtime_r(&time, &local);
  std::ostringstream stream;
  stream << std::put_time(&local, "%Y%m%d_%H%M%S");
  return stream.str();
}

fs::path OutputPath(const ConsoleOptions& options, const std::string& type,
                    uint64_t index) {
  const fs::path path = options.output_directory /
                        (type + "_" + Timestamp() + "_" + std::to_string(index));
  fs::create_directories(path);
  return path;
}

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
  explicit Console(ConsoleOptions options)
      : options_(std::move(options)),
        color_(TerminalColors()),
        session_(BuildInferenceOptions()) {}

  int Run() {
    PrintBanner(color_);
    const auto initialization_started = std::chrono::steady_clock::now();
    PrintInitializationProgress("Starting", initialization_started);
    session_.Initialize([&](const std::string& stage) {
      PrintInitializationProgress(stage, initialization_started);
    });
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

  locateanything::InferenceOptions BuildInferenceOptions() const {
    locateanything::InferenceOptions inference;
    setenv("HB_DNN_USER_DEFINED_L2M_SIZES", options_.l2m_sizes.c_str(), 1);
    inference.vision_model =
        (options_.model_directory / options_.vision_model).string();
    inference.language_model =
        (options_.model_directory / options_.language_model).string();
    inference.embeddings =
        (options_.model_directory / options_.embeddings).string();
    fs::path tokenizer = options_.tokenizer_directory;
    if (tokenizer.empty()) tokenizer = options_.model_directory / "tokenizer";
    if (!fs::is_directory(tokenizer)) tokenizer = "models/tokenizer";
    inference.tokenizer_directory = tokenizer.string();
    inference.generation_mode = options_.generation_mode;
    inference.max_new_tokens = options_.max_new_tokens;
    inference.vision_backend_mask = options_.vision_backend_mask;
    inference.language_backend_mask = options_.language_backend_mask;
    inference.nms_iou = options_.nms_iou;
    return inference;
  }

  void LoadImage(const std::string& value) {
    const fs::path path = fs::absolute(fs::path(value));
    if (!fs::is_regular_file(path) || cv::imread(path.string()).empty()) {
      throw std::runtime_error("image not found or unreadable: " + path.string());
    }
    media_ = {MediaKind::kImage, path};
    std::cout << color_.green << "Image loaded  " << color_.reset << path << '\n';
  }

  void LoadVideo(const std::string& value) {
    const fs::path path = fs::absolute(fs::path(value));
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

  void RunImage(const fs::path& path, const std::string& command) {
    const cv::Mat image = cv::imread(path.string());
    if (image.empty()) throw std::runtime_error("failed to read image: " + path.string());
    locateanything::InferenceOutputOptions output_options;
    output_options.render_annotated = true;
    output_options.serialize_json = true;
    locateanything::InferenceOutput output =
        session_.Infer(image, command, 1, output_options);
    const fs::path directory = OutputPath(options_, "image", ++request_index_);
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

  void RunVideo(const fs::path& path, const std::string& command) {
    cv::VideoCapture video(path.string());
    if (!video.isOpened()) throw std::runtime_error("failed to open video: " + path.string());
    const int width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
    const int64_t expected = static_cast<int64_t>(video.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = video.get(cv::CAP_PROP_FPS);
    if (!(fps > 0.0)) fps = 25.0;

    const fs::path directory = OutputPath(options_, "video", ++request_index_);
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
  uint64_t request_index_ = 0;
  std::string loading_stage_;
};

}  // namespace

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
