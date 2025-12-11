# Breeze-ASR-25 Support Branch

This branch adds support for **Breeze-ASR-25** (MediaTek's Taiwanese Mandarin ASR model) with CoreML ANE optimization to whisper.cpp.

## 🎯 Key Features

- ✅ **Breeze-ASR-25 Model Support** - Full integration of MediaTek's Taiwanese Mandarin model
- ✅ **CoreML ANE Optimization** - Hardware acceleration using Apple Neural Engine
- ✅ **Float16 Output Support** - Enhanced performance with FP16 precision
- ✅ **XPC Service Compatibility** - Fixed Metal library loading in XPC environment
- ✅ **Embedded Metal Shaders** - Bundled Metal shaders for reliable deployment

---

## 📊 Differences from Upstream

**Branch**: `breeze-asr-25-support`  
**Base**: `upstream/master` (ggml-org/whisper.cpp)  
**Commits ahead**: 5  
**Files changed**: 20 files (+34,312, -68)

### Commit History

1. **`c814386d`** - Initial Breeze-ASR-25 support with CoreML ANE optimization
2. **`6cc74d2a`** - Add CoreML Breeze-ASR-25 model support and ggml components
3. **`4447c71b`** - Merge upstream/master (preserve BF16 support)
4. **`08c289dd`** - 🔥 **Critical Fix**: Metal loading in XPC + CoreML Float16 output
5. **`00cccd91`** - Add Metal embed files, remove large model files

---

## 🔧 Core Modifications

### Critical Changes

#### 1. **Metal Library Loading Fix** (`ggml-metal-device.m`)
- **Problem**: Metal library failed to load in XPC Service environment
- **Solution**: Added embedded Metal shader support
- **Impact**: Enables Turbo model to work in sandboxed XPC processes
- **Lines changed**: 98

#### 2. **CoreML Float16 Support** (`whisper-encoder.mm`)
- **Feature**: Support for Float16 output format
- **Benefit**: Better ANE performance and compatibility
- **Target**: Optimized for Breeze-ASR-25 model
- **Lines changed**: 109

#### 3. **Whisper Core Enhancements** (`whisper.cpp`)
- **Updates**: Breeze-ASR-25 specific processing logic
- **Lines changed**: 176

#### 4. **CoreML Conversion Script** (`convert-whisper-to-coreml.py`)
- **Feature**: Support for Breeze-ASR-25 model conversion
- **Optimization**: Enhanced ANE compatibility
- **Lines changed**: 161

### New Files

| File | Purpose | Size |
|------|---------|------|
| `ggml-metal-embed.c` | Embedded Metal shaders | 33,742 lines |
| `ggml-metal-embed.h` | Metal shader header | - |
| `include/ggml-*.h` | GGML API headers | - |
| `models/conversion.log` | Model conversion log | - |

### Modified Files

- `ggml/src/ggml-cpu/ggml-cpu.cpp` - CPU optimizations
- `ggml/src/ggml-metal/ggml-metal.metal` - Metal shader code
- `include/whisper.h` - API extensions (6 lines)

### Removed Files

- Xcode workspace files (cleanup)
- Large CoreML model files (moved to HuggingFace)

---

## 🚀 Usage

### Building with Breeze-ASR-25

```bash
# Clone this fork
git clone https://github.com/sheep52031/whisper.cpp.git
cd whisper.cpp
git checkout breeze-asr-25-support

# Build with CoreML support
cmake -B build -DWHISPER_COREML=1
cmake --build build
```

### Model Files

Large model files are hosted on HuggingFace:
- Breeze-ASR-25 CoreML models
- Pre-compiled `.mlmodelc` packages

---

## 🔄 Syncing with Upstream

To keep this branch up-to-date with upstream:

```bash
# Fetch latest from upstream
git fetch upstream

# Check differences
git log upstream/master..HEAD --oneline

# Merge upstream changes (carefully handle conflicts)
git merge upstream/master
```

### Merge Strategy

- **Preserve**: BF16 support, Metal embed files, CoreML Float16
- **Review carefully**: Changes to `whisper.cpp`, `ggml-metal-device.m`
- **Test after merge**: XPC Service functionality, ANE acceleration

---

## 📝 Maintenance Notes

### Regular Tasks

1. **Weekly**: Check upstream for critical updates
2. **Monthly**: Sync with upstream/master
3. **After sync**: Test Breeze-ASR-25 model inference
4. **Update**: Metal embed files if shader changes

### Known Considerations

- ⚠️ Large Metal embed file (33K+ lines) may cause merge conflicts
- ⚠️ CoreML-specific changes may not align with upstream direction
- ✅ BF16 support is preserved across merges

---

## 🤝 Contributing

This fork is maintained for **Breeze-ASR-25** support. If you find issues:

1. Check if the issue exists in upstream first
2. For Breeze-ASR-25 specific issues, open an issue in this repo
3. For general whisper.cpp issues, report to upstream

---

## 📄 License

Same as upstream whisper.cpp (MIT License)

---

## 🔗 Links

- **Upstream**: https://github.com/ggml-org/whisper.cpp
- **Breeze-ASR-25**: MediaTek Research
- **Models**: HuggingFace (link TBD)

---

**Last Updated**: 2025-12-11  
**Upstream Sync**: 2025-12-11 (commit `9f5ed26e`)
