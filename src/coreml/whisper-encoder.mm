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
    if (@available(macOS 13.0, iOS 16.0, *)) {
        config.computeUnits = MLComputeUnitsAll; // Reverted to standard
        
        // 🔧 DEBUG: Log compute units
        NSLog(@"[CoreML] Loading model with MLComputeUnitsAll");  // ⚡ 優先使用 ANE 加速 (針對 Breeze-ASR-25 優化)
    } else {
        // Fallback for older OS versions if needed, or just use default
        config.computeUnits = MLComputeUnitsAll;
        NSLog(@"[CoreML] Loading model with MLComputeUnitsAll (fallback)");
    }

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
        // 3. Create feature provider
        // The standard model uses "logmel_data" as input
        NSError *error = nil;
        MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{ @"logmel_data" : inMultiArray } error:&error];
        
        if (!provider) {
            NSLog(@"whisper_coreml_encode: error creating feature provider: %@", error);
            return;
        }

        // 4. Prediction
        whisper_encoder_impl *model = (__bridge whisper_encoder_impl*) ctx->data;
        id<MLFeatureProvider> outputFeatures = [model.model predictionFromFeatures:provider error:&error];
        if (!outputFeatures) {
            NSLog(@"whisper_coreml_encode: error during prediction: %@", error);
            if (constraint) {
                NSLog(@"[CoreML] Model expected shape: %@", expectedShape);
            }
            NSLog(@"[CoreML] Provided MLMultiArray shape: %@", inputShape);
            NSLog(@"[CoreML] Provided strides: %@", @[@(n_ctx*n_mel), @(n_ctx), @1]);
            return;
        }

        // 5. Get output
        // The standard model uses "output" or "encoder_output" depending on conversion?
        // Upstream uses "output" usually, but let's check what my script generated.
        // My script generated "output".
        MLFeatureValue *outputValue = [outputFeatures featureValueForName:@"output"];
        if (!outputValue || outputValue.type != MLFeatureTypeMultiArray) {
            NSLog(@"whisper_coreml_encode: invalid output feature (expected 'encoder_output')");
            return;
        }

        MLMultiArray *outputArray = outputValue.multiArrayValue;
        NSLog(@"whisper_coreml_encode output shape: %@", outputArray.shape);

        if (outputArray.dataType == MLMultiArrayDataTypeFloat32) {
            const float *outputPtr = (const float *)outputArray.dataPointer;
            memcpy(out, outputPtr, outputArray.count * sizeof(float));
            
            // Debug log for first few samples
            NSMutableString *samples = [NSMutableString string];
            double sum = 0.0;
            for (NSInteger i = 0; i < MIN(10, outputArray.count); ++i) {
                float value = outputPtr[i];
                sum += value;
                [samples appendFormat:@"%0.4f ", value];
            }
            NSLog(@"whisper_coreml_encode output (Float32) first 10: %@ (sum=%0.4f)", samples, sum);
            
        } else if (outputArray.dataType == MLMultiArrayDataTypeFloat16) {
            // Convert Float16 to Float32
            // _Float16 is available in C11/C++17, but for ObjC/C we might need to be careful.
            // CoreML uses standard float16.
            // We can use __fp16 if supported, or just cast if compiler supports it.
            // Assuming standard ARM64 environment which supports __fp16.
            
            const __fp16 *outputPtr = (const __fp16 *)outputArray.dataPointer;
            for (NSInteger i = 0; i < outputArray.count; ++i) {
                out[i] = (float)outputPtr[i];
            }
            
            // Debug log
            NSMutableString *samples = [NSMutableString string];
            double sum = 0.0;
            for (NSInteger i = 0; i < MIN(10, outputArray.count); ++i) {
                float value = (float)outputPtr[i];
                sum += value;
                [samples appendFormat:@"%0.4f ", value];
            }
            NSLog(@"whisper_coreml_encode output (Float16 converted) first 10: %@ (sum=%0.4f)", samples, sum);
            
        } else if (outputArray.dataType == MLMultiArrayDataTypeDouble) {
            const double *outputPtr = (const double *)outputArray.dataPointer;
            for (NSInteger i = 0; i < outputArray.count; ++i) {
                out[i] = (float)outputPtr[i];
            }
             NSLog(@"whisper_coreml_encode output (Double converted)");
        } else {
             NSLog(@"whisper_coreml_encode: unsupported data type %ld", (long)outputArray.dataType);
        }
    }
}

#if __cplusplus
}
#endif
