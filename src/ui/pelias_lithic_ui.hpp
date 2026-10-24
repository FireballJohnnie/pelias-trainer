// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// ui/pelias_lithic_ui.hpp: the "lithic to voxel" Dear ImGui drawing layer of the Pelias Trainer.
//
// What it is: a few hundred lines of ImDrawList calls that give the whole trainer one look (slate canvas, recessed
// wells, stone bevels, cyan voxels), a spear-shaped progress bar, a chiselled Greek lettering renderer for headers and
// milestones, a warning window with four actions, and a telemetry/log panel.
//
// Cost model (all of it is enforced by the types below, and measured by tests/ui):
//   * Nothing is allocated per frame. Particles live in fixed std::array pools; text is formatted into stack buffers.
//   * Worker threads never block the interface and the interface never blocks them: telemetry is std::atomic, the log
//     is a single-producer/single-consumer ring, warnings and commands travel through a lock-free seqlock.
//   * The spear is two pre-baked textures (stone half, glowing half) cut at the progress point with a ragged edge, so
//     the detail of the artwork costs two textured quads per strip, not drawing code. Without the textures a procedural
//     low-poly spear is drawn instead.
//
// Typography: Tier 1 (all interactive text) is the host's own ImGui font. Tier 2 (page titles, milestones, the percent
// readout) is the procedural chiselled Greek lettering of DrawGreekLapidaryText. The letter C is drawn as an angular '<'.
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>
#include <vector>

#include "imgui.h"

// ------------------------------------------------------------------------------------------------------------------
// Palette
// ------------------------------------------------------------------------------------------------------------------
namespace pelias::palette {

constexpr ImU32 Rgba(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a = 255) { return IM_COL32(r, g, b, a); }

/// Multiplies the alpha of a packed colour by a factor in [0, 1].
constexpr ImU32 WithAlpha(ImU32 c, float factor) {
    const float a = static_cast<float>((c >> IM_COL32_A_SHIFT) & 0xFFu) * (factor < 0.f ? 0.f : (factor > 1.f ? 1.f : factor));
    return (c & ~IM_COL32_A_MASK) | (static_cast<ImU32>(a + 0.5f) << IM_COL32_A_SHIFT);
}

inline constexpr ImU32 MatteSlate = Rgba(0x26, 0x2A, 0x31);       // canvas
inline constexpr ImU32 DeepSlateWell = Rgba(0x1D, 0x20, 0x26);    // recessed wells, inputs, the modal tablet
inline constexpr ImU32 MarbleLight = Rgba(0xC6, 0xCC, 0xD4);      // top/left bevels, ridge highlights
inline constexpr ImU32 MarbleMid = Rgba(0x89, 0x90, 0x99);        // stone facet body
inline constexpr ImU32 MarbleShadow = Rgba(0x4E, 0x53, 0x5B);     // bottom/right bevels
inline constexpr ImU32 FissureDark = Rgba(0x20, 0x23, 0x28);      // crack lines at the seam
inline constexpr ImU32 WeatheredTaupe = Rgba(0x8F, 0x85, 0x78);   // warm stone, tumbling shards
inline constexpr ImU32 CoreBloomCyan = Rgba(0xC4, 0xF7, 0xFF);    // bright seam core
inline constexpr ImU32 AegeanVoxelCyan = Rgba(0x6C, 0xD4, 0xFF);  // interface accents, log pips
inline constexpr ImU32 HoloWireframeCyan = Rgba(0x38, 0x98, 0xEC, 115);  // 45 % alpha
inline constexpr ImU32 VoxelHaze = Rgba(0x38, 0x98, 0xEC, 46);           // 18 % alpha
inline constexpr ImU32 OlympianLightningYellow = Rgba(0xFF, 0xD6, 0x00);
inline constexpr ImU32 LightningCore = Rgba(0xFF, 0xFD, 0xE7);
inline constexpr ImU32 ThermalAmber = Rgba(0xFF, 0xB3, 0x00);
inline constexpr ImU32 CrimsonFault = Rgba(0xE5, 0x39, 0x35);
inline constexpr ImU32 DeadBasalt = Rgba(0x3E, 0x43, 0x4C);
inline constexpr ImU32 TextPrimary = Rgba(0xE6, 0xE9, 0xED);
inline constexpr ImU32 TextSecondary = Rgba(0x8E, 0x95, 0x9E);

/// MEASURED in the artwork (pelias-spear-0.png, the 10 and 50 percent images): the voxel squares and hologram dots have a
/// hue near 185 degrees, a little greener than AegeanVoxelCyan (199 degrees). The spear's particles use this so they match
/// the baked glowing texture; the interface accents keep AegeanVoxelCyan as specified.
inline constexpr ImU32 ArtVoxelCyan = Rgba(0x7F, 0xE3, 0xEA);

}  // namespace pelias::palette

namespace pelias::ui {

// ------------------------------------------------------------------------------------------------------------------
// States and shared (cross-thread) data
// ------------------------------------------------------------------------------------------------------------------
enum class TransmuteState : std::uint8_t { IDLE, RUNNING, WARNING_PAUSED, COMPLETE, FAULT_STOPPED };
enum class LogLevel : std::uint8_t { Info, Warning, Fault };
enum class UiCommand : std::uint8_t { None, Confirm, Deny, TryAgain };

/// Copies a string into a fixed char array, always NUL-terminated, truncating if needed. Returns the stored length.
template <std::size_t N>
std::size_t CopyText(std::array<char, N>& dst, std::string_view src) {
    const std::size_t n = src.size() < N - 1 ? src.size() : N - 1;
    for (std::size_t i = 0; i < n; ++i) dst[i] = src[i];
    dst[n] = '\0';
    return n;
}

struct LogLine {
    LogLevel level{LogLevel::Info};
    std::array<char, 112> text{};
};

/// Single-producer / single-consumer ring of log lines. The producer is a worker thread, the consumer the interface
/// thread. Push never blocks: when the ring is full the line is dropped and `dropped` is incremented.
template <std::size_t N>
class SpscLogRing {
    static_assert((N & (N - 1)) == 0, "N must be a power of two");

  public:
    bool Push(LogLevel level, std::string_view text) {
        const std::uint32_t h = head_.load(std::memory_order_relaxed);
        const std::uint32_t t = tail_.load(std::memory_order_acquire);
        if (h - t >= N) {
            dropped.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        LogLine& l = slots_[h & (N - 1)];
        l.level = level;
        CopyText(l.text, text);
        head_.store(h + 1, std::memory_order_release);
        return true;
    }
    bool Pop(LogLine& out) {
        const std::uint32_t t = tail_.load(std::memory_order_relaxed);
        const std::uint32_t h = head_.load(std::memory_order_acquire);
        if (t == h) return false;
        out = slots_[t & (N - 1)];
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }
    std::atomic<std::uint32_t> dropped{0};

  private:
    std::array<LogLine, N> slots_{};
    std::atomic<std::uint32_t> head_{0};
    std::atomic<std::uint32_t> tail_{0};
};

/// One writer, any number of readers, no locks. The payload is stored in 64-bit atomic words so that a reader that
/// overlaps a write sees a version change and retries instead of a data race. T must be trivially copyable.
template <typename T>
class SeqlockChannel {
    static_assert(std::is_trivially_copyable_v<T>, "payload must be trivially copyable");
    static constexpr std::size_t kWords = (sizeof(T) + 7) / 8;

  public:
    void Write(const T& v) {
        std::uint64_t buf[kWords] = {};
        std::memcpy(buf, &v, sizeof(T));
        const std::uint32_t s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1, std::memory_order_relaxed);  // odd: write in progress
        std::atomic_thread_fence(std::memory_order_release);
        for (std::size_t i = 0; i < kWords; ++i) words_[i].store(buf[i], std::memory_order_relaxed);
        seq_.store(s + 2, std::memory_order_release);  // even: stable
    }
    /// Returns false only if the channel was never written. Retries (bounded) while a write is in flight.
    bool Read(T& out) const {
        for (int attempt = 0; attempt < 64; ++attempt) {
            const std::uint32_t s1 = seq_.load(std::memory_order_acquire);
            if (s1 == 0) return false;
            if (s1 & 1u) continue;
            std::uint64_t buf[kWords];
            for (std::size_t i = 0; i < kWords; ++i) buf[i] = words_[i].load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq_.load(std::memory_order_relaxed) == s1) {
                std::memcpy(&out, buf, sizeof(T));
                return true;
            }
        }
        return false;
    }
    std::uint32_t Version() const { return seq_.load(std::memory_order_acquire); }

  private:
    std::atomic<std::uint32_t> seq_{0};
    std::array<std::atomic<std::uint64_t>, kWords> words_{};
};

/// What a worker publishes when something needs the user (all text is copied in; nothing is referenced).
struct WarningInfo {
    std::array<char, 160> what{};      // what the warning is
    std::array<char, 160> fix{};       // proposed direction to fix it
    std::array<char, 48> parameter{};  // the setting that would change, e.g. "trainer.micro_batch_size"
    std::array<char, 32> current{};    // its current value as text
    std::array<char, 32> proposed{};   // the proposed value as text
    float step{1.0f};                  // what the - and + buttons add or subtract (supplied by the worker)
    std::uint32_t training_step{0};    // for the log; the worker puts the step number in `what` as well
};

/// What the interface hands back after the user chooses.
struct CommandMsg {
    UiCommand command{UiCommand::None};
    std::uint32_t serial{0};
    std::array<char, 64> value{};  // the value to apply (the proposed one, or the user's override)
};

/// Everything the trainer threads and the interface thread exchange. All members are lock-free.
struct SharedState {
    std::atomic<float> fraction{0.f};
    std::atomic<float> loss{0.f};
    std::atomic<float> throughput{0.f};  // tokens per second
    std::atomic<float> vram_used_gib{0.f};
    std::atomic<float> vram_total_gib{0.f};
    std::atomic<float> gpu_temp_c{0.f};
    std::atomic<std::uint32_t> epoch{0};
    std::atomic<std::uint32_t> step{0};
    std::atomic<TransmuteState> state{TransmuteState::IDLE};
    SpscLogRing<256> log;
    SeqlockChannel<WarningInfo> warning;
    SeqlockChannel<CommandMsg> command;
};
static_assert(std::atomic<float>::is_always_lock_free, "float atomics must be lock-free");
static_assert(std::atomic<std::uint32_t>::is_always_lock_free, "u32 atomics must be lock-free");
static_assert(std::atomic<TransmuteState>::is_always_lock_free, "state atomics must be lock-free");
static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "u64 atomics must be lock-free");

/// Worker-side helper: publish a warning and pause. The interface shows the window; the worker polls `command`.
void PublishWarning(SharedState& s, const WarningInfo& w);

// ------------------------------------------------------------------------------------------------------------------
// Assets (loaded once at start-up; this is the only place that allocates)
// ------------------------------------------------------------------------------------------------------------------
struct RgbaImage {
    int w{0};
    int h{0};
    std::vector<std::uint8_t> pixels;  // w * h * 4, straight alpha
};

/// Creates a GPU texture from RGBA8 pixels and returns its ImTextureID. Supplied by the host (any backend).
using CreateTextureFn = ImTextureID (*)(const std::uint8_t* rgba, int width, int height, void* user);

struct Texture {
    ImTextureID id{};
    int w{0};
    int h{0};
    bool Valid() const { return w > 0 && h > 0; }
};

/// Mapping between a progress fraction and the horizontal texture coordinate of the baked spear (spear_layout.json).
struct SpearLayout {
    float u0{0.0125f};  // texture coordinate of the pommel's left end
    float u1{0.9875f};  // texture coordinate of the spearhead's tip
};

struct Assets {
    Texture logo;
    Texture spear_stone;
    Texture spear_holo;
    SpearLayout layout;
    /// The logo is shown through this window of the picture (face and name, without the empty margin). The file is not changed.
    float logo_u0{0.2539f}, logo_v0{0.1307f}, logo_u1{0.7472f}, logo_v1{0.8693f};
    bool HasSpearTextures() const { return spear_stone.Valid() && spear_holo.Valid(); }
    bool HasLogo() const { return logo.Valid(); }
    /// Bytes of RGBA texture memory the three textures occupy (a lower bound for the GPU memory they use).
    std::size_t TextureBytes() const {
        return static_cast<std::size_t>(logo.w) * logo.h * 4 + static_cast<std::size_t>(spear_stone.w) * spear_stone.h * 4 +
               static_cast<std::size_t>(spear_holo.w) * spear_holo.h * 4;
    }
};

/// Decodes a PNG into RGBA8. If max_side > 0 and the picture is larger, it is box-filtered down so its longer side is
/// max_side. Needs PELIAS_UI_WITH_STB_IMAGE; without it this returns false and fills `error`.
bool LoadRgbaFile(const char* path, int max_side, RgbaImage& out, char* error, std::size_t error_cap);

/// Reads u0/u1 from spear_layout.json (a small scan, not a JSON parser). Returns false if either is missing.
bool ParseSpearLayout(const char* json_text, SpearLayout& out);

/// Loads <folder>/pelias-logo.png, spear_stone.png, spear_holo.png and spear_layout.json and uploads them through the
/// callback. Whatever loads is kept; whatever fails leaves its Texture invalid and appends a line to `error`.
/// Returns true when all three textures loaded. logo_max_side limits the logo texture (0 = as provided).
bool LoadAssets(const char* folder, CreateTextureFn create, void* user, int logo_max_side, Assets& out, char* error,
                std::size_t error_cap);

// ------------------------------------------------------------------------------------------------------------------
// Configuration
// ------------------------------------------------------------------------------------------------------------------
struct LithicConfig {
    /// Called when the user presses a [?] help button; the host opens the .txt file. May be null.
    void (*open_help_file)(const char* path, void* user) = nullptr;
    void* open_help_user = nullptr;
    float animation_hz = 30.0f;        // particle simulation rate; drawing may be faster, the simulation is not
    bool feather_logo_edges = true;    // blend the logo picture's own background into the canvas at its four edges
    /// Optional warning colouring of the telemetry panel. 0 = off. The library has no built-in limits.
    float vram_warn_fraction = 0.f;
    float temp_warn_c = 0.f;
    float temp_fault_c = 0.f;
    float splash_seconds = 2.2f;
};

// ------------------------------------------------------------------------------------------------------------------
// Chiselled lettering
// ------------------------------------------------------------------------------------------------------------------
enum class TextAlign : std::uint8_t { Left, Center, Right };

struct ChiselStyle {
    ImU32 light = palette::MarbleLight;    // top-left bevel
    ImU32 mid = palette::MarbleMid;        // stone body
    ImU32 shadow = palette::MarbleShadow;  // lower-right facet
};

/// Width and height in pixels of `text` drawn at the given capital height. UTF-8; letters A-Z and a-z (drawn as capitals),
/// digits, Pi, and  % . / : - ! + < > ( ) _ and the space.
ImVec2 MeasureGreekLapidaryText(std::string_view text, float cap_height_px);
void DrawGreekLapidaryText(ImDrawList* dl, ImVec2 pos, std::string_view text, float cap_height_px,
                           TextAlign align = TextAlign::Left, const ChiselStyle& style = {}, float alpha = 1.0f);
/// "0%" to "100%" in chiselled numerals.
void DrawChiseledPercent(ImDrawList* dl, ImVec2 pos, int percent, float cap_height_px, TextAlign align = TextAlign::Left,
                         float alpha = 1.0f);

// ------------------------------------------------------------------------------------------------------------------
// Spear progress bar
// ------------------------------------------------------------------------------------------------------------------
struct LithicShard {
    float x{}, y{};      // reference pixels from the seam (reference spear width = 1000 px)
    float vx{}, vy{};
    float rot{}, vrot{};
    float size{};
    float life{}, max_life{1.f};
    std::uint8_t sides{3};
    std::uint8_t tone{0};
};
struct VoxelParticle {
    float x{}, y{};
    float vx{}, vy{};
    float size{};
    float life{}, max_life{1.f};
};
struct DustMote {
    float x{}, y{};  // 0..1 across the spear
    float vx{}, vy{};
    float size{};
    float phase{};
};

/// All animation state of one spear. Fixed pools, no heap. Advance() runs the simulation at a fixed rate.
struct SpearAnimator {
    std::array<LithicShard, 18> shards{};
    std::array<VoxelParticle, 40> voxels{};
    std::array<DustMote, 12> dust{};
    std::uint32_t rng{0x9E3779B9u};
    float accumulator{0.f};
    float time{0.f};         // simulation time, frozen with the particles
    float shown_fraction{0.f};
    bool seeded{false};
    float flash_clock{0.f};  // lightning flicker clock, runs while paused
    std::uint32_t bolt_seed{1};

    void Seed(std::uint32_t seed);
    /// dt = real seconds since the last call. Particles and the eased fraction advance in fixed steps of 1/hz.
    void Advance(float dt, float target_fraction, TransmuteState state, float hz);
};

/// Total height the bar needs for a given width (spear + readout).
float SpearBarHeight(float width, const Assets& assets);

/// Draws the spear, the seam effects and the chiselled readout inside pos..pos+size. The spear is scaled to the width.
void DrawPelianSpearProgressBar(ImDrawList* dl, const ImVec2& pos, const ImVec2& size, float fraction, TransmuteState state,
                                SpearAnimator& anim, const Assets& assets, double time_s);

/// A jagged two-layer lightning bolt from `top` to `bottom`. width_px is the outer width; alpha scales everything.
void DrawLightningBolt(ImDrawList* dl, ImVec2 top, ImVec2 bottom, std::uint32_t seed, float width_px, float alpha);

// ------------------------------------------------------------------------------------------------------------------
// The interface object
// ------------------------------------------------------------------------------------------------------------------
class LithicUi;

struct PageHooks {
    /// Draws the body of page `stage` (0 dataset, 1 config, 2 train, 3 evaluate) inside the main window's content region.
    void (*draw_page)(LithicUi& ui, int stage, void* user) = nullptr;
    void* user = nullptr;
};

enum class ButtonKind : std::uint8_t { Neutral, Primary, Danger };

class LithicUi {
  public:
    static constexpr int kStageCount = 4;

    bool Init(const LithicConfig& config, const Assets& assets, SharedState* shared);

    /// Call once per frame, after ImGui::NewFrame(): drains the log, notices new warnings, advances the animation.
    void BeginFrame(float dt_seconds);

    /// The whole trainer window: header with stage titles and navigation, the page body, the footnote bar, and (when a
    /// worker has paused for a warning) the warning window. pos/size are screen coordinates.
    void DrawMainWindow(const ImVec2& pos, const ImVec2& size, const PageHooks& hooks);

    /// Launch splash ("<Pi>ELIAS // INITIALIZING..."). Returns true while it is still showing.
    bool DrawLaunchSplash(const ImVec2& pos, const ImVec2& size);

    /// Two-column metric readouts above the recessed log. Call inside the page body.
    void DrawBorderlessTrainerTelemetry(float log_height);

    /// The warning window, drawn over `area`. DrawMainWindow calls this when the state is WARNING_PAUSED.
    void DrawLithicWarningModal(const ImVec2& area_pos, const ImVec2& area_size);

    /// A beveled stone button. Returns true when clicked.
    bool Button(const char* label, ImVec2 size, ButtonKind kind = ButtonKind::Neutral, bool toggled = false);

    /// A setting label with the dotted cyan underline: hovering shows a bubble and puts the one-sentence text in the
    /// footnote bar. `plain` and `detail` may be null. Nothing is copied: the strings must outlive the frame.
    void SettingLabel(const char* label, const char* short_text, const char* plain = nullptr, const char* detail = nullptr);

    /// A small [?] button that asks the host to open a help .txt file.
    void HelpButton(const char* id, const char* file_path);

    /// Header: logo, chiselled page title, the four step indicators. Returns the stage the user clicked, or -1.
    int DrawStageHeader(float width);

    // --- state the host may read or set ---
    SharedState& shared() { return *shared_; }
    const Assets& assets() const { return assets_; }
    const LithicConfig& config() const { return config_; }
    SpearAnimator& spear() { return anim_; }
    int stage() const { return stage_; }
    void SetStage(int s) { stage_ = s < 0 ? 0 : (s >= kStageCount ? kStageCount - 1 : s); }
    bool show_details() const { return show_details_; }
    void set_show_details(bool v) { show_details_ = v; }
    float time() const { return static_cast<float>(clock_); }

  private:
    friend struct LithicUiTestAccess;
    void DrainLog();
    void DrawFootnoteBar(float width);
    void DrawLogWell(float height);
    bool ConsumeWarningTransition();

    LithicConfig config_{};
    Assets assets_{};
    SharedState* shared_{nullptr};
    SpearAnimator anim_{};
    double clock_{0.0};
    int stage_{2};
    bool show_details_{false};

    // log display ring (fixed)
    std::array<LogLine, 256> log_lines_{};
    std::uint32_t log_count_{0};
    std::uint32_t log_start_{0};
    bool log_follow_{true};

    // footnote
    const char* footnote_text_{nullptr};
    const char* footnote_detail_{nullptr};
    double footnote_time_{-100.0};

    // warning window
    TransmuteState prev_state_{TransmuteState::IDLE};
    WarningInfo warning_{};
    std::array<char, 64> edit_value_{};
    bool drawer_open_{false};
    double warning_start_{0.0};
    std::uint32_t command_serial_{0};
    float splash_clock_{0.f};
};

// ------------------------------------------------------------------------------------------------------------------
// Theme and the demo
// ------------------------------------------------------------------------------------------------------------------
/// Sets ImGui's colours and metrics to the slate / recessed-well / cyan look (square corners, no borders).
void ApplyLithicTheme(ImGuiStyle& style);

/// Tells the demo where the resources/ui folder is and how to create textures. Call before the first
/// RenderPeliasTrainerDemoWindow(); without it the demo runs with the procedural spear and a text-only header.
void ConfigurePeliasDemo(const char* resource_folder, CreateTextureFn create, void* user);

/// One call per frame, between ImGui::NewFrame() and ImGui::Render(): shows the splash, then the four pages with the
/// spear slider, the completion state and the interactive warning window.
void RenderPeliasTrainerDemoWindow();

/// For tests: moves the demo to a state without user input. stage 0..3, fraction 0..1, state, and whether the splash is skipped.
struct DemoOverride {
    int stage{2};
    float fraction{0.f};
    TransmuteState state{TransmuteState::RUNNING};
    bool skip_splash{true};
    bool open_drawer{false};
    bool details{false};
};
void SetPeliasDemoOverride(const DemoOverride& o);
/// For tests: the demo's UI object (valid after the first RenderPeliasTrainerDemoWindow call).
LithicUi* GetPeliasDemoUi();

}  // namespace pelias::ui
