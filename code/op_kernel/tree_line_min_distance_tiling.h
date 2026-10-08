#pragma once
#include <cstdint>
struct TreeLineMinDistanceTilingData {
    uint32_t b, n, m;
    uint32_t mode;        // 0 segment lanes, 1 point lanes, 2 scalar, 3 blocked
    uint32_t chunks;      // point chunks per batch
    uint32_t chunkPoints; // points owned by one chunk
    uint32_t tilePoints;  // points resident in one vector tile
    uint32_t group;       // points carried by one blocked call
};
