#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "metrics.hpp"
#include "processing/image.hpp"
#include "processing/prompt.hpp"

namespace locateanything {
namespace {

TEST(PromptBuilderTest, BuildsDetectionCategories) {
  const Prompt prompt = PromptBuilder{}.Build("/detect person, car");
  EXPECT_EQ(prompt.task, "object_detection");
  EXPECT_NE(prompt.normalized.find("person</c>car"), std::string::npos);
  size_t visual_tokens = 0;
  for (size_t offset = 0;
       (offset = prompt.model_input.find("<IMG_CONTEXT>", offset)) !=
       std::string::npos;
       offset += std::string("<IMG_CONTEXT>").size()) {
    ++visual_tokens;
  }
  EXPECT_EQ(visual_tokens, 576U);
}

TEST(PromptBuilderTest, RejectsEmptyAndIncompleteCommands) {
  EXPECT_THROW(PromptBuilder{}.Build(""), std::invalid_argument);
  EXPECT_THROW(PromptBuilder{}.Build("/detect"), std::invalid_argument);
  EXPECT_THROW(PromptBuilder{}.Build("/unknown target"), std::invalid_argument);
}

TEST(ImageInputTest, ConvertsTightlyPackedNv12) {
  std::vector<uint8_t> nv12(12, 128U);
  const cv::Mat bgr = Nv12ToBgr(nv12.data(), nv12.size(), 4, 2, 4);
  EXPECT_FALSE(bgr.empty());
  EXPECT_EQ(bgr.cols, 4);
  EXPECT_EQ(bgr.rows, 2);
  EXPECT_EQ(bgr.channels(), 3);
}

TEST(ImageInputTest, RejectsInvalidNv12GeometryAndSize) {
  std::vector<uint8_t> nv12(12, 128U);
  EXPECT_THROW(Nv12ToBgr(nv12.data(), nv12.size(), 3, 2, 3),
               std::runtime_error);
  EXPECT_THROW(Nv12ToBgr(nv12.data(), 8, 4, 2, 4), std::runtime_error);
}

TEST(StageTimingTest, UsesRecordedStageBounds) {
  StageTiming timing;
  timing.start_ms = 4.25;
  timing.end_ms = 9.75;
  EXPECT_DOUBLE_EQ(timing.DurationMs(), 5.5);
  timing.end_ms = 1.0;
  EXPECT_DOUBLE_EQ(timing.DurationMs(), 0.0);
}

}  // namespace
}  // namespace locateanything
