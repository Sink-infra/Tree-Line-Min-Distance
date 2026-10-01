#include "aclnn/opdev/platform.h"
#include "mstx/ms_tools_ext.h"
#include "platform/platform_info.h"
#include <map>
// Execute contiguous ND TreeLineMinDistance tensors through ACLNN.
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn_tree_line_min_distance.h"

#include <algorithm>
#include <cstdint>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

namespace fs = std::experimental::filesystem;

namespace {
std::uintptr_t Gcd(std::uintptr_t a, std::uintptr_t b)
{
    while (b) { const auto remainder = a % b; a = b; b = remainder; }
    return a;
}

int lastCode = 0;
std::string lastStage;
constexpr std::size_t kGuardBytes = 512;
constexpr std::size_t kStorageAlignment = 4096;
constexpr std::uint8_t kGuardValue = 0xA5;

void Check(int status, const char *stage)
{
    if (status != ACL_SUCCESS) {
        lastCode = status; lastStage = stage;
        throw std::runtime_error(std::string(stage) + " status=" + std::to_string(status));
    }
}

std::vector<std::uint8_t> Read(const fs::path &path, std::size_t bytes)
{
    if (fs::file_size(path) != bytes) {
        throw std::runtime_error("wrong input byte count: " + path.string());
    }
    std::ifstream stream(path.string(), std::ios::binary);
    std::vector<std::uint8_t> result(bytes);
    if (!stream.read(reinterpret_cast<char *>(result.data()), bytes)) {
        throw std::runtime_error("input read failed: " + path.string());
    }
    return result;
}

void Save(const fs::path &path, const std::vector<std::uint8_t> &bytes)
{
    std::ofstream stream(path.string(), std::ios::binary | std::ios::trunc);
    const auto saved = stream.rdbuf()->sputn(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    stream.close();
    if (!stream || saved != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error("output save failed: " + path.string());
    }
}

std::int64_t Integer(const std::string &value, std::int64_t low, std::int64_t high)
{
    std::size_t used = 0;
    const auto number = std::stoll(value, &used);
    if (used != value.size() || number < low || number > high) {
        throw std::runtime_error("integer outside allowed range: " + value);
    }
    return number;
}

struct CacheBudget {
    std::string profile, nativeSoc;
    std::uint64_t nativeBytes=0, appliedBytes=0;
};

CacheBudget ConfigureCacheProfile(const std::string &profile)
{
    CacheBudget result;
    result.profile=profile;
    const char *soc=aclrtGetSocName();
    if (!soc) throw std::runtime_error("native device description unavailable");
    result.nativeSoc=soc;
    auto &manager=fe::PlatformInfoManager::Instance();
    if (manager.InitializePlatformInfo()!=0) throw std::runtime_error("platform configuration unavailable");
    fe::PlatFormInfos target;
    fe::OptionalInfos optional;
    if (profile!="native" && manager.GetPlatformInfos(profile,target,optional)!=0)
        throw std::runtime_error("target cache profile unavailable");
    auto *native=op::GetCurrentPlatformInfo().GetPlatformInfos();
    if (!native) throw std::runtime_error("runtime platform unavailable");
    native->GetLocalMemSize(fe::LocalMemType::L2,result.nativeBytes);
    result.appliedBytes=result.nativeBytes;
    if (profile=="native") return result;
    std::map<std::string,std::string> currentSpec,targetSpec;
    if (!native->GetPlatformResWithLock("SoCInfo",currentSpec)
        || !target.GetPlatformResWithLock("SoCInfo",targetSpec))
        throw std::runtime_error("cache specification unavailable");
    std::uint64_t requested=0;
    target.GetLocalMemSize(fe::LocalMemType::L2,requested);
    if (!requested || requested>result.nativeBytes)
        throw std::runtime_error("cache budget exceeds native capacity");
    currentSpec["l2_size"]=targetSpec.at("l2_size");
    native->SetPlatformResWithLock("SoCInfo",currentSpec);
    native->GetLocalMemSize(fe::LocalMemType::L2,result.appliedBytes);
    if (result.appliedBytes!=requested) throw std::runtime_error("cache profile differs from the request");
    return result;
}

class Session {
public:
    explicit Session(std::int32_t device, std::uint32_t vectorCores, const std::string &cacheProfile) : device_(device)
    {
        try {
            Check(aclInit(nullptr), "aclInit");
            initialized_ = true;
            Check(aclrtSetDevice(device_), "aclrtSetDevice");
            selected_ = true;
            Check(aclrtCreateContext(&context_, device_), "aclrtCreateContext");
            Check(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
            Check(aclrtCreateStream(&stream_), "aclrtCreateStream");
            if (vectorCores) {
                Check(aclrtSetStreamResLimit(stream_, ACL_RT_DEV_RES_VECTOR_CORE, vectorCores), "aclrtSetStreamResLimit");
                Check(aclrtUseStreamResInCurrentThread(stream_), "aclrtUseStreamResInCurrentThread");
                limited_ = true;
                Check(aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &vectorCores_), "aclrtGetResInCurrentThread");
                if (vectorCores_ != vectorCores) throw std::runtime_error("vector resource limit differs from the request");
            }
            cache_=ConfigureCacheProfile(cacheProfile);
        } catch (...) {
            Reset();
            throw;
        }
    }
    ~Session() { Reset(); }
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    aclrtStream Stream() const { return stream_; }
    std::uint32_t VectorCores() const { return vectorCores_; }
    const CacheBudget &Cache() const { return cache_; }

private:
    CacheBudget cache_;
    void Reset() noexcept
    {
        if (limited_) aclrtUnuseStreamResInCurrentThread(stream_);
        if (stream_) aclrtDestroyStream(stream_);
        if (context_) aclrtDestroyContext(context_);
        if (selected_) aclrtResetDevice(device_);
        if (initialized_) aclFinalize();
        stream_ = nullptr;
        context_ = nullptr;
        selected_ = initialized_ = false;
    }
    std::int32_t device_;
    std::uint32_t vectorCores_ = 0;
    bool initialized_ = false, selected_ = false, limited_ = false;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
};

class Buffer {
public:
    explicit Buffer(std::size_t bytes) : bytes_(bytes)
    {
        if (bytes_) Check(aclrtMalloc(&data_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }
    ~Buffer() { if (data_) aclrtFree(data_); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    void *Data() const { return data_; }
    std::size_t Bytes() const { return bytes_; }

private:
    void *data_ = nullptr;
    std::size_t bytes_;
};

class Tensor {
public:
    Tensor(std::vector<std::int64_t> shape, aclDataType dtype, std::size_t itemSize)
        : shape_(std::move(shape)), bytes_(Elements(shape_) * itemSize), buffer_(bytes_ + 2 * kGuardBytes + kStorageAlignment - 1)
    {
        const auto start = reinterpret_cast<std::uintptr_t>(buffer_.Data()) + kGuardBytes;
        offset_ = kGuardBytes + (kStorageAlignment - start % kStorageAlignment) % kStorageAlignment;
        if (Alignment() != kStorageAlignment) throw std::runtime_error("device tensor alignment differs from storage configuration");
        Check(aclrtMemset(buffer_.Data(), buffer_.Bytes(), kGuardValue, buffer_.Bytes()), "initialize tensor guards");
        // Device output storage starts from deterministic zeros.
        Check(aclrtMemset(Data(), bytes_, 0, bytes_), "initialize output storage");
        std::vector<std::int64_t> strides(shape_.size(), 1);
        for (std::size_t i = shape_.size() - 1; i > 0; --i) strides[i - 1] = strides[i] * shape_[i];
        tensor_ = aclCreateTensor(shape_.data(), shape_.size(), dtype, strides.data(), 0,
                                  ACL_FORMAT_ND, shape_.data(), shape_.size(), Data());
        if (!tensor_) throw std::runtime_error("aclCreateTensor failed");
    }
    ~Tensor() { if (tensor_) aclDestroyTensor(tensor_); }
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
    aclTensor *Get() const { return tensor_; }
    std::size_t Alignment() const
    {
        return Gcd(reinterpret_cast<std::uintptr_t>(Data()), static_cast<std::uintptr_t>(kStorageAlignment));
    }
    void Load(const std::vector<std::uint8_t> &bytes)
    {
        if (bytes.size() != bytes_) throw std::runtime_error("input tensor size mismatch");
        Check(aclrtMemcpy(Data(), bytes_, bytes.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE), "copy input to device");
    }
    void InitializeOutput(bool nan) const
    { Check(aclrtMemset(Data(), bytes_, nan ? 0xFF : 0, bytes_), "initialize output storage"); }
    std::vector<std::uint8_t> CopyAndCheck(bool &guardsOk) const
    {
        std::vector<std::uint8_t> allocation(buffer_.Bytes());
        Check(aclrtMemcpy(allocation.data(), allocation.size(), buffer_.Data(), buffer_.Bytes(),
                          ACL_MEMCPY_DEVICE_TO_HOST), "copy tensor and guards to host");
        const auto guarded = [](std::vector<std::uint8_t>::const_iterator begin,
                                std::vector<std::uint8_t>::const_iterator end) {
            return std::all_of(begin, end, [](std::uint8_t value) { return value == kGuardValue; });
        };
        if (!guarded(allocation.begin(), allocation.begin() + offset_)
            || !guarded(allocation.begin() + offset_ + bytes_, allocation.end())) {
            guardsOk = false;
        }
        return {allocation.begin() + offset_, allocation.begin() + offset_ + bytes_};
    }

private:
    static std::size_t Elements(const std::vector<std::int64_t> &shape)
    {
        std::size_t size = 1;
        for (auto dim : shape) size *= static_cast<std::size_t>(dim);
        return size;
    }
    void *Data() const { return static_cast<std::uint8_t *>(buffer_.Data()) + offset_; }
    std::vector<std::int64_t> shape_;
    std::size_t bytes_, offset_ = 0;
    Buffer buffer_;
    aclTensor *tensor_ = nullptr;
};

std::string LibraryPath()
{
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find("libcust_opapi.so") != std::string::npos) {
            const auto start = line.find('/');
            if (start != std::string::npos)
                return fs::canonical(line.substr(start)).string();
        }
    }
    throw std::runtime_error("operator library is not mapped");
}

void PrimeHostContext(const fs::path &folder)
{
    std::ifstream config((folder / "case.conf").string());
    std::map<std::string, std::string> fields;
    std::string line;
    while (std::getline(config, line)) {
        const auto split = line.find('=');
        if (split == std::string::npos || !fields.emplace(line.substr(0, split), line.substr(split + 1)).second)
            throw std::runtime_error("invalid or duplicate context configuration field");
    }
    if (!config.eof() || fields.size() != 4) throw std::runtime_error("invalid context configuration");
    const auto b = Integer(fields.at("b"), 1, 8);
    const auto n = Integer(fields.at("n"), 1, 16384);
    const auto m = Integer(fields.at("m"), 1, 1024);
    const auto dtype = fields.at("dtype");
    if (dtype != "float32" && dtype != "float16") throw std::runtime_error("unsupported context dtype");
    const auto type = dtype == "float32" ? ACL_FLOAT : ACL_FLOAT16;
    const std::size_t itemSize = dtype == "float32" ? 4 : 2;
    Tensor treePoints({b, n, 3}, type, itemSize), lineSegments({b, m, 2, 3}, type, itemSize);
    Tensor distance({b, n}, type, itemSize), index({b, n}, ACL_INT32, 4);
    std::uint64_t workspaceBytes = 0;
    aclOpExecutor *executor = nullptr;
    Check(aclnnTreeLineMinDistanceGetWorkspaceSize(treePoints.Get(), lineSegments.Get(), distance.Get(),
                                                 index.Get(), &workspaceBytes, &executor), "context workspace query");
}

void Execute(const fs::path &folder, const fs::path &output,
             Session &session, int warmup, int repeat, bool performance, bool nanOutput)
{
    std::ifstream config((folder / "case.conf").string());
    std::map<std::string, std::string> fields;
    std::string line;
    while (std::getline(config, line)) {
        const auto split = line.find('=');
        if (split == std::string::npos || !fields.emplace(line.substr(0, split), line.substr(split + 1)).second)
            throw std::runtime_error("invalid or duplicate fixture configuration field");
    }
    if (!config.eof() || fields.size() != 4) throw std::runtime_error("invalid fixture configuration");
    const auto b = Integer(fields.at("b"), 1, 8);
    const auto n = Integer(fields.at("n"), 1, 16384);
    const auto m = Integer(fields.at("m"), 1, 1024);
    const auto dtype = fields.at("dtype");
    if (dtype != "float32" && dtype != "float16") throw std::runtime_error("unsupported execution dtype");
    const aclDataType type = dtype == "float32" ? ACL_FLOAT : ACL_FLOAT16;
    const std::size_t itemSize = dtype == "float32" ? 4 : 2;
    const auto points = Read(folder / "tree_points.bin", b * n * 3 * itemSize);
    const auto segments = Read(folder / "line_segments.bin", b * m * 6 * itemSize);
    Tensor treePoints({b, n, 3}, type, itemSize), lineSegments({b, m, 2, 3}, type, itemSize);
    Tensor distance({b, n}, type, itemSize), index({b, n}, ACL_INT32, 4);
    treePoints.Load(points);
    lineSegments.Load(segments);
    if (nanOutput) { distance.InitializeOutput(true); index.InitializeOutput(true); }
    std::uint64_t workspaceBytes = 0;
    aclOpExecutor *executor = nullptr;
    Check(aclnnTreeLineMinDistanceGetWorkspaceSize(treePoints.Get(), lineSegments.Get(), distance.Get(),
                                                 index.Get(), &workspaceBytes, &executor), "workspace query");
    Buffer workspace(static_cast<std::size_t>(workspaceBytes));
    bool guardsOk = true, inputsUnchanged = true;
    try {
        for (int iteration = 0; iteration < warmup + repeat; ++iteration) {
            if (iteration) {
                Check(aclnnTreeLineMinDistanceGetWorkspaceSize(treePoints.Get(), lineSegments.Get(), distance.Get(),
                                                             index.Get(), &workspaceBytes, &executor), "workspace query");
                if (workspaceBytes > workspace.Bytes()) throw std::runtime_error("workspace grew");
            }
            if (!performance) { distance.InitializeOutput(nanOutput); index.InitializeOutput(nanOutput); }
            Check(aclnnTreeLineMinDistance(workspace.Data(), workspaceBytes, executor, session.Stream()), "kernel launch");
            Check(aclrtSynchronizeStream(session.Stream()), "stream synchronize");
            if ((!performance && iteration == warmup) || iteration == warmup + repeat - 1) {
                const auto actualDistance = distance.CopyAndCheck(guardsOk);
                const auto actualIndex = index.CopyAndCheck(guardsOk);
                const bool treeUnchanged = treePoints.CopyAndCheck(guardsOk) == points;
                const bool linesUnchanged = lineSegments.CopyAndCheck(guardsOk) == segments;
                inputsUnchanged = treeUnchanged && linesUnchanged && inputsUnchanged;
                if (!performance && iteration == warmup) {
                    Save(output / "first_min_distance.bin", actualDistance);
                    Save(output / "first_nearest_segment_idx.bin", actualIndex);
                }
                if (iteration == warmup + repeat - 1) {
                    Save(output / "min_distance.bin", actualDistance);
                    Save(output / "nearest_segment_idx.bin", actualIndex);
                }
            }
        }
    } catch (...) {
        aclrtSynchronizeStream(session.Stream());
        throw;
    }
    std::ofstream((output / "execution.txt").string())
        << "status=executed\napi_code=0\nguards_ok=" << guardsOk
        << "\ninputs_unchanged=" << inputsUnchanged << "\nrepeat=" << repeat
        << "\nalignment_bytes=" << Gcd(Gcd(treePoints.Alignment(), lineSegments.Alignment()),
                                              Gcd(distance.Alignment(), index.Alignment()))
        << "\nguard_bytes=" << kGuardBytes << "\nmode=" << (performance ? "performance" : "correctness")
        << "\noutput_init=" << (nanOutput ? "nan" : "zero")
        << "\nvector_cores=" << session.VectorCores()
        << "\ncache_profile=" << session.Cache().profile
        << "\nnative_soc=" << session.Cache().nativeSoc
        << "\nnative_l2_cache_bytes=" << session.Cache().nativeBytes
        << "\nl2_cache_budget_bytes=" << session.Cache().appliedBytes
        << "\ndtype=" << dtype << "\nlibrary=" << LibraryPath() << "\n";
}
} // namespace

namespace {
void Failure(const fs::path &output, const std::exception &error)
{
    std::ofstream((output / "execution.txt").string())
        << "status=Runtime Error\napi_code=" << lastCode << "\nstage=" << lastStage
        << "\ndetail=" << error.what() << "\n";
    std::ofstream((output / "case-exit.txt").string()) << "1\n";
}

std::vector<std::string> Names(const fs::path &path)
{
    std::ifstream input(path.string());
    std::vector<std::string> names;
    std::map<std::string, bool> seen;
    std::string name;
    while (std::getline(input, name)) {
        if (name.empty() || !std::all_of(name.begin(), name.end(), [](unsigned char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
                    || (c >= '0' && c <= '9') || c == '_' || c == '-';
            }) || !seen.emplace(name, true).second)
            throw std::runtime_error("invalid or duplicate case name");
        names.push_back(name);
    }
    if (!input.eof() || names.empty()) throw std::runtime_error("empty or unreadable case list");
    return names;
}
} // namespace

int main(int argc, char **argv)
{
    fs::path fixture, output, caseList, contextCatalog, profileList;
    std::vector<std::string> names;
    std::size_t completed = 0;
    bool stopped = false;
    try {
        const std::uint16_t endian = 1;
        if (*reinterpret_cast<const std::uint8_t *>(&endian) != 1)
            throw std::runtime_error("little-endian host required");
        std::int32_t device = 0;
        int warmup = 0, repeat = 1;
        std::uint32_t vectorCores = 0;
        std::string mode = "correctness", outputInit = "zero", cacheProfile = "native";
        std::map<std::string, std::string> options;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc || !options.emplace(argv[i], argv[i + 1]).second)
                throw std::runtime_error("invalid or duplicate option");
        }
        for (const auto &option : options) {
            const auto &key = option.first;
            const auto &value = option.second;
            if (key == "--fixture") fixture = value;
            else if (key == "--output") output = value;
            else if (key == "--case-list") caseList = value;
            else if (key == "--context-catalog") contextCatalog = value;
            else if (key == "--profile-list") profileList = value;
            else if (key == "--mode") mode = value;
            else if (key == "--output-init") outputInit = value;
            else if (key == "--cache-profile") cacheProfile = value;
            else if (key == "--device") device = Integer(value, 0, std::numeric_limits<std::int32_t>::max());
            else if (key == "--warmup") warmup = Integer(value, 0, 500);
            else if (key == "--repeat") repeat = Integer(value, 1, 5000);
            else if (key == "--vector-cores") vectorCores = Integer(value, 0, 40);
            else throw std::runtime_error("unknown option: " + key);
        }
        if (fixture.empty() || output.empty() || (mode != "correctness" && mode != "performance")
            || (outputInit != "zero" && outputInit != "nan")
            || (cacheProfile != "native" && cacheProfile != "Ascend910B4"))
            throw std::runtime_error("fixture, output, and a valid mode required");
        names = caseList.empty() ? std::vector<std::string>{""} : Names(caseList);
        if (!profileList.empty() && (mode != "performance" || warmup != 0 || repeat != 5))
            throw std::runtime_error("profile selection requires five performance calls and native warm-up");
        const auto profileNames = profileList.empty() ? names : Names(profileList);
        for (const auto &name : profileNames)
            if (std::find(names.begin(), names.end(), name) == names.end())
                throw std::runtime_error("profile Case missing from execution order");
        const auto contextNames = contextCatalog.empty() ? names : Names(contextCatalog / "order.txt");
        std::size_t contextIndex = 0;
        if (!contextCatalog.empty()) {
            for (const auto &name : names) {
                while (contextIndex < contextNames.size() && contextNames[contextIndex] != name) ++contextIndex;
                if (contextIndex == contextNames.size()) throw std::runtime_error("selected cases are not an ordered catalogue subset");
                ++contextIndex;
            }
            contextIndex = 0;
        }
        for (const auto &name : names) fs::create_directories(output / name);
        Session session(device, vectorCores, cacheProfile);
        for (const auto &name : names) {
            const auto destination = output / name;
            if (stopped) {
                std::ofstream((destination / "execution.txt").string())
                    << "status=Skipped\nreason=preceding runtime failure\n";
                std::ofstream((destination / "case-exit.txt").string()) << "3\n";
                continue;
            }
            lastCode = 0; lastStage.clear();
            try {
                if (!contextCatalog.empty()) {
                    while (contextNames[contextIndex] != name) {
                        PrimeHostContext(contextCatalog / contextNames[contextIndex]);
                        ++contextIndex;
                    }
                    ++contextIndex;
                }
                if (mode == "performance") std::clog << "<CaseReplay> " << name << " " << getpid() << std::endl;
                const bool sampled = std::find(profileNames.begin(), profileNames.end(), name) != profileNames.end();
                const bool ranged = !profileList.empty() && sampled;
                const auto range = ranged ? mstxRangeStartA("selected", nullptr) : 0;
                try {
                    Execute(fixture / name, destination, session, warmup, sampled ? repeat : repeat * 7,
                            mode == "performance", outputInit == "nan");
                } catch (...) {
                    if (ranged) mstxRangeEnd(range);
                    throw;
                }
                if (ranged) mstxRangeEnd(range);
                std::ofstream((destination / "case-exit.txt").string()) << "0\n";
                ++completed;
            } catch (const std::exception &error) {
                Failure(destination, error);
                stopped = true;
            }
        }
        std::ofstream((output / "batch.txt").string())
            << "status=completed\ncase_count=" << names.size() << "\ncompleted_cases=" << completed
            << "\nstopped=" << stopped << "\n";
        return stopped ? 1 : 0;
    } catch (const std::exception &error) {
        if (!output.empty()) {
            bool first = true;
            for (const auto &name : names) {
                if (first) { Failure(output / name, error); first = false; }
                else {
                    std::ofstream((output / name / "execution.txt").string()) << "status=Skipped\nreason=session initialization failure\n";
                    std::ofstream((output / name / "case-exit.txt").string()) << "3\n";
                }
            }
            std::ofstream((output / "batch.txt").string()) << "status=Runtime Error\napi_code=" << lastCode
                << "\ndetail=" << error.what() << "\n";
        }
        return 1;
    }
}
