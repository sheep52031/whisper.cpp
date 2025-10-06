#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "whisper-encoder.h"
#import "whisper-encoder-impl.h"

#import <CoreML/CoreML.h>

#include <stdlib.h>

#if __cplusplus
extern "C" {
#endif

struct whisper_coreml_context {
    const void * data;
};

struct whisper_coreml_context * whisper_coreml_init(const char * path_model) {
    NSString * path_model_str = [[NSString alloc] initWithUTF8String:path_model];

    NSURL * url_model = [NSURL fileURLWithPath: path_model_str];

    // select which device to run the Core ML model on
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    // config.computeUnits = MLComputeUnitsCPUAndGPU;
    // config.computeUnits = MLComputeUnitsAll;  // 原始配置（混合執行）
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;  // ⚡ 優先使用 ANE 加速 (針對 Breeze-ASR-25 優化)

    const void * data = CFBridgingRetain([[whisper_encoder_impl alloc] initWithContentsOfURL:url_model configuration:config error:nil]);

    if (data == NULL) {
        return NULL;
    }

    whisper_coreml_context * ctx = new whisper_coreml_context;

    ctx->data = data;

    return ctx;
}

void whisper_coreml_free(struct whisper_coreml_context * ctx) {
    CFRelease(ctx->data);
    delete ctx;
}

void whisper_coreml_encode(
        const whisper_coreml_context * ctx,
                             int64_t   n_ctx,
                             int64_t   n_mel,
                               float * mel,
                               float * out) {
    NSLog(@"whisper_coreml_encode input shape: 1 x %lld x %lld", n_mel, n_ctx);
    NSArray<NSNumber *> *inputShape = @[@1, @(n_mel), @(n_ctx)];
    // 🔧 FIX: Breeze-ASR-25 CoreML 模型使用 "mel" 作為輸入名稱
    MLMultiArrayConstraint *constraint = [(__bridge id)ctx->data model].modelDescription.inputDescriptionsByName[@"mel"].multiArrayConstraint;
    NSArray<NSNumber *> *expectedShape = constraint.shape;

    if (![inputShape isEqualToArray:expectedShape]) {
        NSLog(@"[CoreML] Input shape mismatch: expected %@, actual %@", expectedShape, inputShape);
    }

    MLMultiArray * inMultiArray = [
        [MLMultiArray alloc] initWithDataPointer: mel
                                           shape: inputShape
                                        dataType: MLMultiArrayDataTypeFloat32
                                         strides: @[@(n_ctx*n_mel), @(n_ctx), @1]
                                     deallocator: nil
                                           error: nil
    ];

    @autoreleasepool {
        NSError *error = nil;
        // 🔧 FIX: 使用 MLDictionaryFeatureProvider 直接指定正確的 feature 名稱 "mel"
        // 不依賴自動生成的 predictionFromLogmel_data: 方法（它硬編碼了錯誤的名稱）
        MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{@"mel": inMultiArray} error:&error];

        if (error) {
            NSLog(@"whisper_coreml_encode feature provider error: %@", error);
            return;
        }

        whisper_encoder_impl *model = (__bridge whisper_encoder_impl*) ctx->data;
        id<MLFeatureProvider> outputFeatures = [model.model predictionFromFeatures:provider error:&error];

        if (error) {
            NSLog(@"whisper_coreml_encode prediction error: %@", error);
            if (constraint) {
                NSLog(@"[CoreML] Model expected shape: %@", expectedShape);
            }
            NSLog(@"[CoreML] Provided MLMultiArray shape: %@", inputShape);
            NSLog(@"[CoreML] Provided strides: %@", @[@(n_ctx*n_mel), @(n_ctx), @1]);
            return;
        }

        // 🔧 FIX: Breeze-ASR-25 CoreML 模型輸出名稱為 "encoder_output"
        MLFeatureValue *outputValue = [outputFeatures featureValueForName:@"encoder_output"];
        if (!outputValue || outputValue.type != MLFeatureTypeMultiArray) {
            NSLog(@"whisper_coreml_encode: invalid output feature (expected 'encoder_output')");
            return;
        }

        MLMultiArray *outputArray = outputValue.multiArrayValue;
        NSLog(@"whisper_coreml_encode output shape: %@", outputArray.shape);

        const float *outputPtr = (const float *)outputArray.dataPointer;
        double sum = 0.0;
        const NSInteger sampleCount = MIN(10, outputArray.count);
        NSMutableString *samples = [NSMutableString string];
        for (NSInteger i = 0; i < sampleCount; ++i) {
            float value = outputPtr[i];
            sum += value;
            [samples appendFormat:@"%0.4f ", value];
        }
        NSLog(@"whisper_coreml_encode output first %ld samples: %@ (sum first=%0.4f)", (long)sampleCount, samples, sum);

        memcpy(out, outputPtr, outputArray.count * sizeof(float));
    }
}

#if __cplusplus
}
#endif
