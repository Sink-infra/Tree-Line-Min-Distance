#pragma once

// UB offsets in FP32 elements, including the conversion and reduction scratch.
struct HoistedPointGroupLayout {
    static constexpr uint32_t kPoints = 120;
    static constexpr uint32_t oDx = 9280;
    static constexpr uint32_t oDy = 16960;
    static constexpr uint32_t oDz = 24640;
    static constexpr uint32_t oTt = 32416;
    static constexpr uint32_t oDsq = 9280;
    static constexpr uint32_t oSegOff = 40096;
    static constexpr uint32_t oPtRaw = 41120;
    static constexpr uint32_t oPtFlt = 41504;
    static constexpr uint32_t oPtOff = 41888;
    static constexpr uint32_t oDiv8 = 42016;
    static constexpr uint32_t oPx = 42016;
    static constexpr uint32_t oPy = 42144;
    static constexpr uint32_t oPz = 42272;
    static constexpr uint32_t oPxv = 42400;
    static constexpr uint32_t oPyv = 43360;
    static constexpr uint32_t oPzv = 44320;
    static constexpr uint32_t oPair = 45280;
    static constexpr uint32_t oCmin = 45536;
    static constexpr uint32_t oCidx = 45664;
    static constexpr uint32_t oBest = 45792;
    static constexpr uint32_t oFidx = 45920;
    static constexpr uint32_t oRes = 46048;
    static constexpr uint32_t oRedW = 46048;
    static constexpr uint32_t oDist = 46048;
    static constexpr uint32_t oIdx = 46176;
    static constexpr uint32_t oHalf = 46304;
    static constexpr uint32_t oMask = 46368;
    static constexpr uint32_t oArena = 46432;
};
static_assert(HoistedPointGroupLayout::oDx + 64 * HoistedPointGroupLayout::kPoints <= HoistedPointGroupLayout::oDy);
static_assert(HoistedPointGroupLayout::oDy + 64 * HoistedPointGroupLayout::kPoints <= HoistedPointGroupLayout::oDz);
static_assert(HoistedPointGroupLayout::oDz + 64 * HoistedPointGroupLayout::kPoints <= HoistedPointGroupLayout::oTt);
static_assert(HoistedPointGroupLayout::oTt + 64 * HoistedPointGroupLayout::kPoints <= HoistedPointGroupLayout::oSegOff);
static_assert(HoistedPointGroupLayout::oPxv + 8 * HoistedPointGroupLayout::kPoints <= HoistedPointGroupLayout::oPyv);
static_assert(HoistedPointGroupLayout::oPyv + 8 * HoistedPointGroupLayout::kPoints <= HoistedPointGroupLayout::oPzv);
static_assert(HoistedPointGroupLayout::oPzv + 8 * HoistedPointGroupLayout::kPoints <= HoistedPointGroupLayout::oPair);
static_assert(HoistedPointGroupLayout::oArena * sizeof(float) <= 192 * 1024);
