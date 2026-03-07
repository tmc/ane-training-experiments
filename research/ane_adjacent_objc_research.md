# ANE-Adjacent Objective-C Recon (ipsw)

Generated: 2026-03-03 22:17:00Z

## Scope
Targets requested:
- Espresso (high)
- RemoteCoreML (high)
- MLRuntime (medium-high)
- PrivateMLClient (medium)

Method:
- dyld-cache extraction via `ipsw class-dump` (plain, refs, re, headers)
- dyld ObjC scans via `ipsw dyld objc class|proto|sel`
- dyld Mach-O ObjC extraction via `ipsw dyld macho --objc --objc-refs`
- runtime executables on this host: `mlruntimed`, `PrivateMLClientInferenceProviderService`

## Coverage Summary
```text
## Espresso
file=research/ipsw_ane_adjacent/raw/Espresso.dsc.classdump.txt
lines=2531 interfaces=128 protocols=7 objc_methods=855 selectors=1177

## RemoteCoreML
file=research/ipsw_ane_adjacent/raw/RemoteCoreML.dsc.classdump.txt
lines=247 interfaces=9 protocols=0 objc_methods=118 selectors=130

## MLRuntime
file=research/ipsw_ane_adjacent/raw/MLRuntime.dsc.classdump.txt
lines=505 interfaces=19 protocols=11 objc_methods=151 selectors=205

## PrivateMLClient
file=research/ipsw_ane_adjacent/raw/PrivateMLClient.dsc.classdump.txt
lines=141 interfaces=11 protocols=0 objc_methods=0 selectors=21

## mlruntimed (arm64e)
file=research/ipsw_ane_adjacent/raw/mlruntimed.arm64e.classdump.txt
lines=211 interfaces=6 protocols=4 objc_methods=76

## mlruntimed (x86_64)
file=research/ipsw_ane_adjacent/raw/mlruntimed.x86_64.classdump.txt
lines=211 interfaces=6 protocols=4 objc_methods=76

## PrivateMLClientInferenceProviderService (arm64e)
file=research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.arm64e.classdump.txt
lines=8 interfaces=1 protocols=0 objc_methods=0

## PrivateMLClientInferenceProviderService (x86_64)
file=research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.x86_64.classdump.txt
lines=8 interfaces=1 protocols=0 objc_methods=0

```

## Espresso
### Class/Protocol Inventory
Classes: 128
Protocols: 7

Protocols:
```text
ETDataProvider
ETDataSource
ETTaskContext
ExternalDetectedObject
MPSCNNConvolutionDataSource
NSCopying
NSObject
```

Top classes by ObjC method count:
```text
EspressoFDOverfeatNetwork	54
EspressoImage2Image	46
ETTaskDefinition	36
ETImageDescriptorExtractor	29
ETDataTensor	25
ETTask	24
ETModelDef	21
_ANECompilerAnalytics	20
ETVariable	19
EspressoANEIOSurface	19
ETDataSourceFromFolderData	18
EspressoDCNEspressoOverfeatDetector	17
EspressoDataFrame	17
Espresso_mxnetTools_ImageBinaryRecordReader	16
ETLossConfig	14
EspressoDataFrameExecutor	14
EspressoProfilingLayerInfo	14
EspressoProfilingLayerSupportInfo	14
_ETBufferDataSource	14
EspressoBrickTensorShape	12
EspressoConvolutionWeightsForMPS	12
EspressoDataFrameAttachment	12
EspressoDataFrameStorageExecutorMatchingBufferSet	12
EspressoInnerProductWeightsForMPS	12
EspressoMetalKernelsCache	12
```

Focused interfaces:
```text
=== EspressoANEIOSurface ===
@interface EspressoANEIOSurface : NSObject {
    /* instance variables */
    @"NSDictionary" params_dict;
    {vector<Espresso::ANERuntimeEngine::surface_and_buffer, std::allocator<Espresso::ANERuntimeEngine::surface_and_buffer>>="__begin_"^{surface_and_buffer}"__end_"^{surface_and_buffer}""{?="__cap_"^{surface_and_buffer}}} multiple_buffer_io_surfaces;
    B created_with_lazy_iosurface;
    B ane_surface_use_cvpixelbuffer;
    Q width;
    Q height;
    Q rowBytes;
    @"NSSet" valid_pixel_formats;
    I _pixelFormat;
    {shared_ptr<Espresso::blob<unsigned char, 1>>="__ptr_"^v"__cntrl_"^{__shared_weak_count}} _external_storage_blob_for_aliasing_mem;
}

@property (T{shared_ptr<Espresso::blob<unsigned char, 1>>=^v^{__shared_weak_count}},N,V_external_storage_blob_for_aliasing_mem) external_storage_blob_for_aliasing_mem;
@property (TI,R,N,V_pixelFormat) pixelFormat;

/* instance methods */
-[EspressoANEIOSurface pixelFormat];
-[EspressoANEIOSurface cleanup];
-[EspressoANEIOSurface checkIfMatches:];
-[EspressoANEIOSurface nFrames];
-[EspressoANEIOSurface ane_io_surfaceForMultiBufferFrame:];
-[EspressoANEIOSurface checkIfMatchesIOSurface:];
-[EspressoANEIOSurface createIOSurfaceWithExtraProperties:];
-[EspressoANEIOSurface doNonLazyAllocation:];
-[EspressoANEIOSurface external_storage_blob_for_aliasing_mem];
-[EspressoANEIOSurface initWithIOSurfaceProperties:andPixelFormats:];
-[EspressoANEIOSurface ioSurfaceForMultiBufferFrame:];
-[EspressoANEIOSurface ioSurfaceForMultiBufferFrameNoLazyForTesting:];
-[EspressoANEIOSurface lazilyAutoCreateSurfaceForFrame:];
-[EspressoANEIOSurface metalBufferWithDevice:multiBufferFrame:];
-[EspressoANEIOSurface resizeForMultipleAsyncBuffers:];
-[EspressoANEIOSurface restoreInternalStorage:];
-[EspressoANEIOSurface restoreInternalStorageForAllMultiBufferFrames];
-[EspressoANEIOSurface setExternalStorage:ioSurface:];
-[EspressoANEIOSurface setExternal_storage_blob_for_aliasing_mem:];

@end

=== EspressoContext ===
@interface EspressoContext : NSObject {
    /* instance variables */
    {shared_ptr<Espresso::abstract_context>="__ptr_"^{abstract_context}"__cntrl_"^{__shared_weak_count}} _ctx;
}

@property (T{shared_ptr<Espresso::abstract_context>=^{abstract_context}^{__shared_weak_count}},R,V_ctx) ctx;
@property (Ti,R) platform;

/* instance methods */
-[EspressoContext initWithContext:];
-[EspressoContext initWithPlatform:];
-[EspressoContext platform];
-[EspressoContext ctx];
-[EspressoContext initWithDevice:andWisdomParams:];
-[EspressoContext set_priority:low_priority_max_ms_per_command_buffer:gpu_priority:];
-[EspressoContext initWithNetworkContext:];

@end

=== EspressoNetwork ===
@interface EspressoNetwork : NSObject {
    /* instance variables */
    {shared_ptr<Espresso::net>="__ptr_"^{net}"__cntrl_"^{__shared_weak_count}} _net;
}

@property (T{shared_ptr<Espresso::net>=^{net}^{__shared_weak_count}},R,V_net) net;
@property (TQ,R) layers_size;
@property (T@"EspressoContext",R) ctx;

/* instance methods */
-[EspressoNetwork ctx];
-[EspressoNetwork initWithJSFile:binSerializerId:context:computePath:];
-[EspressoNetwork initWithJSFile:context:computePath:];
-[EspressoNetwork layers_size];
-[EspressoNetwork wipe_layers_blobs];
-[EspressoNetwork net];

@end

=== _ANECompilerAnalytics ===
@interface _ANECompilerAnalytics : NSObject {
    /* instance variables */
    @"NSArray" _procedureAnalytics;
    @"NSData" _analyticsBuffer;
    @"NSNumber" _bufferSizeInBytes;
}

@property (T@"NSData",R,N,V_analyticsBuffer) analyticsBuffer;
@property (T@"NSNumber",R,N,V_bufferSizeInBytes) bufferSizeInBytes;
@property (T@"NSArray",&,N,V_procedureAnalytics) procedureAnalytics;

/* class methods */
+[_ANECompilerAnalytics new];
+[_ANECompilerAnalytics objectWithBuffer:];

/* instance methods */
-[_ANECompilerAnalytics description];
-[_ANECompilerAnalytics init];
-[_ANECompilerAnalytics serialize];
-[_ANECompilerAnalytics analyticsBuffer];
-[_ANECompilerAnalytics initWithBuffer:];
-[_ANECompilerAnalytics getDataValueAt:];
-[_ANECompilerAnalytics bufferSizeInBytes];
-[_ANECompilerAnalytics dataInfoAt:];
-[_ANECompilerAnalytics getBOOLDataValueAt:];
-[_ANECompilerAnalytics groupInfoAt:];
-[_ANECompilerAnalytics layerInfoAt:];
-[_ANECompilerAnalytics offsetTableAt:count:];
-[_ANECompilerAnalytics populateAnalytics];
-[_ANECompilerAnalytics procedureAnalytics];
-[_ANECompilerAnalytics procedureInfoAt:];
-[_ANECompilerAnalytics setProcedureAnalytics:];
-[_ANECompilerAnalytics stringForAnalyticsType:];
-[_ANECompilerAnalytics taskInfoAt:];

@end

=== _ANEAnalyticsLayer ===
@interface _ANEAnalyticsLayer : NSObject {
    /* instance variables */
    @"NSString" _layerName;
    @"NSNumber" _weight;
}

@property (T@"NSString",R,N,V_layerName) layerName;
@property (T@"NSNumber",R,N,V_weight) weight;

/* class methods */
+[_ANEAnalyticsLayer objectWithName:weight:];

/* instance methods */
-[_ANEAnalyticsLayer serialize];
-[_ANEAnalyticsLayer layerName];
-[_ANEAnalyticsLayer weight];
-[_ANEAnalyticsLayer initWithName:weight:];

@end

```

Key selectors sample (ANE/CoreML/model/runtime filtered):
```text
0x1fafc5c88: selected_runtime_engine
0x1fb3492be: setTotal_ane_time_ns:
0x1fa183994: aneSubTypeProductVariant
0x1fabdf8d4: purgeCompiledModel:
0x1fb3491ac: setPreferredDevice:
0x1fafbb419: setTensorNamed:withValue:error:
0x1fb348c1e: setContext_metal:
0x1fa0f80e3: taskInfo
0x1f9e018fc: setStorageMode:
0x1fb346b1c: ane_io_surfaceForMultiBufferFrame:
0x1fafc20ae: initWithNetwork:
0x1fb346723: setDescriptors_file_cache_size:
0x1fb348a2f: setAllowedComputeDevices:
0x1fb349263: setRuntimes:
0x1fb3495ef: tuneNetworksWGWindowSize:
0x1fa4a415d: aneSubTypeVariant
0x1fa035f90: setModel:
0x1fb349616: unloadEntryPointToSymbolAndFileNameMap:device:compilationDescriptor:
0x1f9eb659e: evaluateWithModel:options:request:qos:error:
0x1f9e0d908: cached
0x1fb348a96: setAverage_runtime:
0x1fb3479d4: initWithHeight:Width:NumChannels:Scale:BiasR:BiasG:BiasB:NetworkWantBGR:
0x1fafbf232: encodeToMetalCommandBuffer:inputTensors:outputTensors:
0x1fb348836: retile_and_forward_espresso_network_at_index:net:pyr:
0x1fb347371: external_storage_blob_for_aliasing_mem
0x1fb348a49: setAne_compiler_analytics:
0x1faacb210: initWithDevice:integerScaleFactorX:integerScaleFactorY:
0x1fb3466db: network_at_path
0x1fb347b56: initWithModelDef:
0x1fb348f08: setInferenceModel:
0x1fa528346: initWithDevice:neuronDescriptor:
0x1fb348918: runOnNetwork:
0x1fb348bc8: setCompiler_analytics_file_names:
0x1fb347c7b: initWithNetworkAtPath:contextObjC:platform:computePath:
0x1fb346dcc: buildNetwork
0x1fb346cce: bindInput:fromTensorAttachment:toNetwork:
0x1fb347b91: initWithModelDef:optimizerDef:extractor:needWeightsInitialization:
0x1fb1d91dd: initWithDevice:andWisdomParams:
0x1fb347a82: initWithInferenceNetworkPath:error:
0x1fa0618d0: aneSubType
0x1fb346d17: bindOutputsFromFrame:toNetwork:executionStatus:
0x1f9dfb990: modelURL
0x1fb34896d: saveNetwork:revertToInferenceMode:
0x1fafdcbdd: setCompilerOptions:
0x1fb013770: initWithDevice:kernelSize:
0x1fb2b5712: ane_performance_info
0x1f9e1d110: minimumLinearTextureAlignmentForPixelFormat:
0x1fb346adb: aneUserInitiatedTaskQoS
0x1fb3495e1: tuneNetworks:
0x1fafc58f0: saveNetwork:inplace:error:
0x1fabc7290: initWithDevice:kernelWidth:kernelHeight:strideInPixelsX:strideInPixelsY:
0x1fad9222c: setM_kernelCache:
0x1fb01b08c: aneUserInteractiveTaskQoS
0x1f9e523ea: initWithDevice:
0x1f9e58ef0: initWithDevice:weights:
0x1fb34707d: dataTypeForParameterOfType:fromLayerNamed:
0x1fb3476f2: getValidateNetworkSupportedVersion
0x1fb346ac8: aneRealTimeTaskQoS
0x1fb2bc51c: total_ane_time_ns
0x1fae31343: writeJSONObject:toStream:options:error:
0x1fb347c47: initWithNetworkAtPath:context:platform:computePath:
0x1fb3480ab: modelAtURLWithSourceURL:sourceURL:key:cacheURLIdentifier:
0x1fb34956d: taskInfoAt:
0x1fb348990: saveTrainingNetwork:checkpoint:error:
0x1fb3474ec: forward_cpu_network_at_index:pyr:
0x1fa742390: setCpuCacheMode:
0x1fafc1813: initWithInferenceNetworkPath:inferenceInputs:inferenceOutputs:error:
0x1fa821aa5: taskState
0x1fa2d3610: setNetwork:
0x1fb0ea4a9: setCached:
0x1fb346ca5: bindInput:fromImageAttachment:toNetwork:
0x1fb348225: objectWithID:layers:tasks:
0x1fb349682: unmapIOSurfacesWithModel:request:
0x1fb346ffa: custom_network_path
0x1fb347936: initWithFolder:forImageTensor:andLabelTensor:shuffleBeforeEachEpoch:shuffleRandomSeed:withImagePreprocessParams:
0x1fb0245d6: modelWithNetworkDescription:weights:optionsPlist:
0x1fb347525: fromDict:frameStorage:
0x1fb346af3: aneUtilityTaskQoS
0x1fb01370f: initWithDevice:isSecondarySourceFilter:
0x1fafbefb9: doInferenceOnData:error:
0x1fad19710: metalDevice
0x1f9ea8bba: unloadModel:options:qos:error:
0x1fae08691: newTextureWithDescriptor:iosurface:plane:
0x1fb346ab6: aneDefaultTaskQoS
0x1fb348da9: setExternal_storage_blob_for_aliasing_mem:
0x1fb349579: tensorWithCGImage:
0x1fb3480e5: modelWithCacheURLIdentifier:
0x1faf6b25d: initWithDevice:momentumScale:useNestrovMomentum:optimizerDescriptor:
0x1fb348eef: setInferenceGraphNetPtr:
0x1fb348528: purgeCompiledModel
0x1fb347fab: loadRealTimeModel:options:qos:error:
0x1fb348c4c: setCustom_network_path:
0x1f9eff710: network
0x1fb346b54: applyEntryPointToSymbolAndFileNameMap:device:compilationDescriptor:
0x1fafbdaca: computeOnCPUWithInputTensors:outputTensors:
0x1fb346aa1: aneBackgroundTaskQoS
0x1fb3476d7: getTensorNamed:directBind:
0x1fb348960: saveNetwork:
0x1fb347071: dataStorage
0x1fb348163: networkPointer
0x1fb34681f: bindOutputsFromFrame:toNetwork:
0x1f9e19d77: compiledModelExistsFor:
0x1fb013bf9: initWithDevice:sourceCount:
0x1fb34759f: getFacesFromNetworkResultOriginalWidth:originalHeight:
0x1fb3470db: descriptors_mem_cache_size
0x1fb348957: runtimes
0x1fb34758c: getEspressoNetwork
0x1f9ff6dba: mapIOSurfacesWithModel:request:cacheInference:error:
0x1fb34965b: unloadRealTimeModel:options:qos:error:
0x1fb34663f: average_runtime
0x1fa5dc019: kernelForFunction:cacheString:withConstants:
0x1fb3490b4: setNetworkPointer:
0x1fb348b42: setBreakUpMetalEncoders:
0x1fabc8010: sharedSession
0x1fb3488f0: runInference:outputNames:batchCallback:
0x1fb3487b6: restoreInternalStorage:
0x1fb347cb3: initWithNetworkContext:
0x1fab662ab: m_kernelCache
0x1fb346cf8: bindInputsFromFrame:toNetwork:
0x1fb3490c7: setNetwork_at_path:
0x1fb346f01: context_for_runtime_platform:
0x1fb349324: setWeightsOfInferenceNetworkLoadedFrom:AndSaveTo:error:
0x1fb348a64: setAne_performance_info:
0x1fb013cfe: initWithDevice:windowInX:windowInY:strideInX:strideInY:
0x1fb3467b3: tensorWithPath:
0x1f9fd2f10: dataTaskWithRequest:completionHandler:
0x1fb347141: disableTypeInference
0x1fa5dfaa7: loadModel:options:qos:error:
0x1fad3f4a4: setTaskState:
0x1fb3472ff: executeDataFrameStorage:withNetwork:referenceNetwork:block:blockPrepareForIndex:
0x1fb348c0d: setContextMetal:
0x1fb021022: inMemoryModelWithDescriptor:
0x1fb347a1d: initWithID:layers:tasks:
0x1fb3485a0: reloadOnRuntimePlatform:
0x1fb346b3f: ane_time_per_eval_ns
0x1fb347b68: initWithModelDef:optimizerDef:extractor:
0x1fafd6fb8: compilerOptions
0x1fb34781c: inferenceGraphNetPtr
0x1fae29364: outputStreamToFileAtPath:append:
0x1fb346d47: bindOutputsFromFrame:toNetwork:referenceNetwork:
```

## RemoteCoreML
### Class/Protocol Inventory
Classes: 9
Protocols: 0

Classes:
```text
_MLLog
_MLNetworkHeaderEncoding
_MLNetworkOptions
_MLNetworkPacket
_MLNetworkUtilities
_MLNetworking
_MLRemoteConnection
_MLRemoteCoreMLErrors
_MLServer
```

Top classes by ObjC method count:
```text
_MLNetworkHeaderEncoding	22
_MLServer	20
_MLNetworking	18
_MLRemoteConnection	16
_MLNetworkOptions	14
_MLNetworkPacket	13
_MLNetworkUtilities	8
_MLLog	5
_MLRemoteCoreMLErrors	2
```

Focused interfaces:
```text
=== _MLNetworking ===
@interface _MLNetworking : NSObject {
    /* instance variables */
    B _isClient;
    @"NSObject<OS_nw_connection>" _connection;
    @"NSObject<OS_nw_listener>" _listener;
    @"_MLNetworkOptions" _nwOptions;
    @"NSObject<OS_nw_protocol_stack>" _protocol_stack;
    @"NSObject<OS_nw_parameters>" _parameters;
    @"NSObject<OS_os_log>" _logType;
    @"NSObject<OS_dispatch_queue>" _q;
}

@property (T@"NSObject<OS_nw_connection>",&,N,V_connection) connection;
@property (T@"NSObject<OS_nw_listener>",R,N,V_listener) listener;
@property (T@"_MLNetworkOptions",R,N,V_nwOptions) nwOptions;
@property (T@"NSObject<OS_nw_protocol_stack>",R,N,V_protocol_stack) protocol_stack;
@property (T@"NSObject<OS_nw_parameters>",R,N,V_parameters) parameters;
@property (TB,R,N,V_isClient) isClient;
@property (T@"NSObject<OS_os_log>",R,N,V_logType) logType;
@property (T@"NSObject<OS_dispatch_queue>",R,N,V_q) q;

/* instance methods */
-[_MLNetworking logType];
-[_MLNetworking listener];
-[_MLNetworking stop];
-[_MLNetworking setConnection:];
-[_MLNetworking connection];
-[_MLNetworking isClient];
-[_MLNetworking startConnection];
-[_MLNetworking parameters];
-[_MLNetworking q];
-[_MLNetworking sendData:];
-[_MLNetworking initConnection:];
-[_MLNetworking receiveLoop:];
-[_MLNetworking setListenerReceiveDataCallBack:];
-[_MLNetworking initListener:];
-[_MLNetworking nwOptions];
-[_MLNetworking protocol_stack];
-[_MLNetworking restartConnection];
-[_MLNetworking setReceiveDataCallBack:];

@end

=== _MLNetworkPacket ===
@interface _MLNetworkPacket : NSObject {
    /* instance variables */
    @"NSMutableData" _buffer;
    @"NSMutableData" _doubleBuffer;
    Q _sizeOfPacket;
    Q _command;
}

@property (T@"NSMutableData",&,N,V_buffer) buffer;
@property (T@"NSMutableData",&,N,V_doubleBuffer) doubleBuffer;
@property (TQ,N,V_sizeOfPacket) sizeOfPacket;
@property (TQ,N,V_command) command;

/* instance methods */
-[_MLNetworkPacket setCommand:];
-[_MLNetworkPacket setBuffer:];
-[_MLNetworkPacket buffer];
-[_MLNetworkPacket init];
-[_MLNetworkPacket reset];
-[_MLNetworkPacket command];
-[_MLNetworkPacket setSizeOfPacket:];
-[_MLNetworkPacket sizeOfPacket];
-[_MLNetworkPacket cleanupDoubleBuffer];
-[_MLNetworkPacket doubleBuffer];
-[_MLNetworkPacket resetDoubleBuffer];
-[_MLNetworkPacket resetMetadata];
-[_MLNetworkPacket setDoubleBuffer:];

@end

=== _MLNetworkOptions ===
@interface _MLNetworkOptions : NSObject {
    /* instance variables */
    @"NSMutableDictionary" _networkOptions;
    q _receiveTimeout;
}

@property (T@"NSMutableDictionary",R,N,V_networkOptions) networkOptions;
@property (Tq,R,N,V_receiveTimeout) receiveTimeout;

/* instance methods */
-[_MLNetworkOptions family];
-[_MLNetworkOptions useAWDL];
-[_MLNetworkOptions initWithOptions:];
-[_MLNetworkOptions port];
-[_MLNetworkOptions networkOptions];
-[_MLNetworkOptions localPort];
-[_MLNetworkOptions psk];
-[_MLNetworkOptions localAddr];
-[_MLNetworkOptions networkNameIdentifier];
-[_MLNetworkOptions receiveTimeout];
-[_MLNetworkOptions receiveTimeoutValue];
-[_MLNetworkOptions useBonjour];
-[_MLNetworkOptions useTLS];
-[_MLNetworkOptions useUDP];

@end

=== _MLServer ===
@interface _MLServer : NSObject {
    /* instance variables */
    @"_MLNetworking" _nwObj;
    @"_MLNetworkOptions" _nwOptions;
    @"_MLNetworkPacket" _packet;
    @? _loadFunction;
    @? _predictFunction;
    @? _unLoadFunction;
    @? _textFunction;
    @"NSObject<OS_dispatch_queue>" _q;
}

@property (T@"_MLNetworking",R,N,V_nwObj) nwObj;
@property (T@"_MLNetworkOptions",R,N,V_nwOptions) nwOptions;
@property (T@"_MLNetworkPacket",R,N,V_packet) packet;
@property (T@?,C,N,V_loadFunction) loadFunction;
@property (T@?,C,N,V_predictFunction) predictFunction;
@property (T@?,C,N,V_unLoadFunction) unLoadFunction;
@property (T@?,C,N,V_textFunction) textFunction;
@property (T@"NSObject<OS_dispatch_queue>",R,N,V_q) q;

/* instance methods */
-[_MLServer initWithOptions:];
-[_MLServer start];
-[_MLServer stop];
-[_MLServer q];
-[_MLServer packet];
-[_MLServer predictFunction];
-[_MLServer setTextCommand:];
-[_MLServer setLoadCommand:];
-[_MLServer doReceive:context:isComplete:error:];
-[_MLServer loadFunction];
-[_MLServer nwObj];
-[_MLServer nwOptions];
-[_MLServer setLoadFunction:];
-[_MLServer setPredictCommand:];
-[_MLServer setPredictFunction:];
-[_MLServer setTextFunction:];
-[_MLServer setUnLoadCommand:];
-[_MLServer setUnLoadFunction:];
-[_MLServer textFunction];
-[_MLServer unLoadFunction];

@end

=== _MLRemoteConnection ===
@interface _MLRemoteConnection : NSObject {
    /* instance variables */
    @"NSObject<OS_nw_connection>" _connection;
    @"_MLNetworking" _nwObj;
    @"_MLNetworkOptions" _nwOptions;
    @"_MLNetworkPacket" _packet;
    @"NSMutableData" _outputResult;
    Q _jobCount;
    @"NSObject<OS_dispatch_semaphore>" _semaphore;
    @"NSObject<OS_dispatch_queue>" _q;
}

@property (T@"NSObject<OS_nw_connection>",R,N,V_connection) connection;
@property (T@"_MLNetworking",R,N,V_nwObj) nwObj;
@property (T@"_MLNetworkOptions",R,N,V_nwOptions) nwOptions;
@property (T@"_MLNetworkPacket",R,N,V_packet) packet;
@property (T@"NSMutableData",&,N,V_outputResult) outputResult;
@property (TQ,R,N,V_jobCount) jobCount;
@property (T@"NSObject<OS_dispatch_semaphore>",R,N,V_semaphore) semaphore;
@property (T@"NSObject<OS_dispatch_queue>",R,N,V_q) q;

/* instance methods */
-[_MLRemoteConnection initWithOptions:];
-[_MLRemoteConnection semaphore];
-[_MLRemoteConnection connection];
-[_MLRemoteConnection q];
-[_MLRemoteConnection outputResult];
-[_MLRemoteConnection jobCount];
-[_MLRemoteConnection loadFromURL:options:error:];
-[_MLRemoteConnection packet];
-[_MLRemoteConnection doReceive:context:isComplete:error:];
-[_MLRemoteConnection nwObj];
-[_MLRemoteConnection nwOptions];
-[_MLRemoteConnection predictionFromURL:features:output:options:error:];
-[_MLRemoteConnection send:options:];
-[_MLRemoteConnection sendDataAndWaitForAcknowledgementOrTimeout:];
-[_MLRemoteConnection setOutputResult:];
-[_MLRemoteConnection unloadFromURL:options:error:];

@end

```

Key selectors sample:
```text
0x1fc9e45a7: setPredictFunction:
0x1fc9e4481: predictionFromURL:features:output:options:error:
0x1fc9e4319: doInitNetwork:
0x1fc9e41c1: predictFunction
0x1fc9e445b: networkNameIdentifier
0x1fc9e4211: predictFeature:
0x1faa71cd9: networkOptions
0x1fc9e4594: setPredictCommand:
0x1fc9e441e: isHeaderPredictFeature:
0x1fc95035d: unLoadModel:
0x1faf51ab6: loadModel:
```

## MLRuntime
### Class/Protocol Inventory
Classes: 19
Protocols: 11

Protocols:
```text
MLRExtensionHostProtocol
MLRExtensionRemoteProtocol
MLROnDemandRemoteProtocol
MLROnDemandTaskPerforming
NSCoding
NSExtensionRequestHandling
NSObject
NSSecureCoding
PKModularService
_EXConnectionHandler
_EXMainConnectionHandler
```

Top classes by ObjC method count:
```text
MLRTaskParameters	17
MLRTrialDediscoTaskResult	11
MLROnDemandConnectionHandler	10
MLRTaskAttachments	10
MLRTask	8
NSDictionary	8
MLRServiceClient	7
MLRTaskResult	7
MLRTrialDediscoRecipe	7
MLRDonationManager	6
MLROnDemandPlugin	6
MLRExtensionRemoteContext	4
MLRSandboxExtensionRequest	4
MLRTrialTaskResult	4
MLRExtensionService_Subsystem	3
MLRTrialTask	3
TRIClient	2
DESRecordSet	1
MLRExtensionPrincipalClass	1
```

Focused interfaces:
```text
=== MLRServiceClient ===
@interface MLRServiceClient : NSObject {
    /* instance variables */
    @"NSXPCConnection" _connection;
}

/* class methods */
+[MLRServiceClient sharedConnection];

/* instance methods */
-[MLRServiceClient initWithConnection:];
-[MLRServiceClient dealloc];
-[MLRServiceClient donateJSONResult:identifier:completion:];
-[MLRServiceClient performOnRemoteObjectWithBlock:errorHandler:];
-[MLRServiceClient performOnRemoteObjectWithBlock:isSynchronous:errorHandler:];
-[MLRServiceClient performSynchronouslyOnRemoteObjectWithBlock:errorHandler:];

@end

=== MLRTask ===
@interface MLRTask : NSObject <NSSecureCoding> {
    /* instance variables */
    @"MLRTaskParameters" _parameters;
    @"MLRTaskAttachments" _attachments;
}

@property (T@"MLRTaskParameters",R,N,V_parameters) parameters;
@property (T@"MLRTaskAttachments",R,C,N,V_attachments) attachments;

/* class methods */
+[MLRTask supportsSecureCoding];

/* instance methods */
-[MLRTask attachments];
-[MLRTask description];
-[MLRTask initWithCoder:];
-[MLRTask encodeWithCoder:];
-[MLRTask parameters];
-[MLRTask initWithParametersDict:];
-[MLRTask initWithParameters:attachments:];

@end

=== MLRTaskResult ===
@interface MLRTaskResult : NSObject <NSSecureCoding> {
    /* instance variables */
    @"NSDictionary" _JSONResult;
    @"NSData" _vector;
}

@property (T@"NSDictionary",R,N,V_JSONResult) JSONResult;
@property (T@"NSData",R,N,V_vector) vector;

/* class methods */
+[MLRTaskResult supportsSecureCoding];

/* instance methods */
-[MLRTaskResult description];
-[MLRTaskResult initWithCoder:];
-[MLRTaskResult encodeWithCoder:];
-[MLRTaskResult vector];
-[MLRTaskResult JSONResult];
-[MLRTaskResult initWithJSONResult:unprivatizedVector:];

@end

=== MLROnDemandConnectionHandler ===
@interface MLROnDemandConnectionHandler : NSObject <_EXMainConnectionHandler, MLROnDemandRemoteProtocol> {
    /* instance variables */
    @"NSXPCConnection" _xpcConnection;
    @"<MLROnDemandTaskPerforming>" _pluginPrincipal;
}

@property (T@"NSXPCConnection",W,N,V_xpcConnection) xpcConnection;
@property (T@"<MLROnDemandTaskPerforming>",&,N,V_pluginPrincipal) pluginPrincipal;
@property (T@,R) principalObject;
@property (TQ,R) hash;
@property (T#,R) superclass;
@property (T@"NSString",R,C) description;
@property (T@"NSString",?,R,C) debugDescription;

/* instance methods */
-[MLROnDemandConnectionHandler xpcConnection];
-[MLROnDemandConnectionHandler principalObject];
-[MLROnDemandConnectionHandler setXpcConnection:];
-[MLROnDemandConnectionHandler shouldAcceptXPCConnection:];
-[MLROnDemandConnectionHandler description];
-[MLROnDemandConnectionHandler initWithPrincipalObject:];
-[MLROnDemandConnectionHandler pluginPrincipal];
-[MLROnDemandConnectionHandler performTask:completionHandler:];
-[MLROnDemandConnectionHandler performTaskInternal:completionHandler:];
-[MLROnDemandConnectionHandler setPluginPrincipal:];

@end

=== MLROnDemandPlugin ===
@interface MLROnDemandPlugin : NSObject

/* class methods */
+[MLROnDemandPlugin performTask:forPluginID:completionHandler:];
+[MLROnDemandPlugin _createXPCConnection:error:];
+[MLROnDemandPlugin _locateWithExtensionID:];
+[MLROnDemandPlugin _performTask:forPluginID:isSynchronous:completionHandler:];
+[MLROnDemandPlugin onDemandPluginIDs];
+[MLROnDemandPlugin synchronouslyPerformTask:forPluginID:error:];

@end

=== MLRTaskParameters ===
@interface MLRTaskParameters : NSObject <NSSecureCoding> {
    /* instance variables */
    Q _count;
    @"NSDictionary" _recipeUserInfo;
}

@property (T@"NSDictionary",R,C,N,V_recipeUserInfo) recipeUserInfo;
@property (TQ,R,N,V_count) count;
@property (T@"NSDictionary",R,C,N) dictionaryRepresentation;

/* class methods */
+[MLRTaskParameters supportsSecureCoding];

/* instance methods */
-[MLRTaskParameters count];
-[MLRTaskParameters dictionaryRepresentation];
-[MLRTaskParameters description];
-[MLRTaskParameters objectForKeyedSubscript:];
-[MLRTaskParameters initWithURL:error:];
-[MLRTaskParameters boolValueForKey:defaultValue:];
-[MLRTaskParameters initWithCoder:];
-[MLRTaskParameters encodeWithCoder:];
-[MLRTaskParameters doubleValueForKey:defaultValue:];
-[MLRTaskParameters stringValueForKey:defaultValue:];
-[MLRTaskParameters floatValueForKey:defaultValue:];
-[MLRTaskParameters integerValueForKey:defaultValue:];
-[MLRTaskParameters unsignedIntegerValueForKey:defaultValue:];
-[MLRTaskParameters initWithParametersDict:];
-[MLRTaskParameters recipeUserInfo];
-[MLRTaskParameters initWithDESRecipe:];

@end

```

Key selectors sample:
```text
0x1fa614f10: sendEventEvaluationSessionFinishForBundleID:success:
0x1fc8b2ddb: onDemandPluginIDs
0x1fc8b2e8f: performTask:completionHandler:
0x1fc8b2f5c: synchronouslyPerformTask:forPluginID:error:
0x1fa0ebc9b: setRemoteObjectInterface:
0x1fc8b2a3d: _performTask:forPluginID:isSynchronous:completionHandler:
0x1fc8b2f18: setPluginPrincipal:
0x1fae322e2: beginRequestWithExtensionContext:
0x1fc8b2ad4: dediscoTaskConfig
0x1fc8b2d8a: mlr_stringValueForKey:defaultValue:
0x1fc8b2cce: mlr_clientWithMLRTask:
0x1fc8b2c99: mlrDediscoMetadata
0x1fb53819f: performTask:forPluginID:completionHandler:
0x1fc8b2d24: mlr_floatValueForKey:defaultValue:
0x1fa9c2610: initForPlugInKit
0x1fa692f22: _extensionAuxiliaryVendorProtocol
0x1fc8b2ef9: recordSetForTask:
0x1fada5de2: hasOnDemandLaunchEntitlement:
0x1fc8b2dae: mlr_unsignedIntegerValueForKey:defaultValue:
0x1fc8b2cac: mlr_boolValueForKey:defaultValue:
0x1fc8b2d6c: mlr_namespaceNameWithMLRTask:
0x1fc8b29e5: pluginPrincipal
0x1fa4ab6bc: _extensionAuxiliaryHostProtocol
0x1fc8b2d00: mlr_doubleValueForKey:defaultValue:
0x1fc8b2a11: _locateWithExtensionID:
0x1fa912b10: sendEventEvaluationSessionStartForBundleID:
0x1fc8b2e55: performSynchronouslyOnRemoteObjectWithBlock:errorHandler:
0x1f9f38510: remoteObjectProxyWithErrorHandler:
0x1fa4203c7: initForPlugInKitWithOptions:
0x1fc8b2d47: mlr_integerValueForKey:defaultValue:
0x1fc8b2ded: performOnRemoteObjectWithBlock:errorHandler:
0x1fa767210: extensionPointIdentifierQuery:
0x1fc8b2e1a: performOnRemoteObjectWithBlock:isSynchronous:errorHandler:
0x1fadea04c: performSelector:
0x1fadea05d: performSelector:withObject:
0x1faa0fa90: initWithMachServiceName:options:
0x1fc8b2ec1: performTaskInternal:completionHandler:
0x1fc8b2f2c: submitForTask:error:
0x1fadea079: performSelector:withObject:withObject:
0x1fa3fac10: synchronousRemoteObjectProxyWithErrorHandler:
0x1fc8b2ce5: mlr_dictionaryValueForKey:
0x1fc8b2eae: performTask:error:
0x1fc8b2b95: initWithInputItems:listenerEndpoint:contextUUID:plugin:
```

## PrivateMLClient
### Class/Protocol Inventory
Classes: 11
Protocols: 0

Classes (Swift-mangled ObjC metadata):
```text
_TtC15PrivateMLClient11ImageParser
_TtC15PrivateMLClient15PrivateMLClient
_TtC15PrivateMLClient20PrivateMLClientAlert
_TtC15PrivateMLClient24TransparencyReporterImpl
_TtC15PrivateMLClient26TransparencyReporterLogger
_TtC15PrivateMLClient29PrivateMLClientSessionManager
_TtC15PrivateMLClient32PrivateMLClientConnectionManager
_TtC15PrivateMLClient37PrivateMLClientCloudComputeConnection
_TtCV15PrivateMLClient39Apple_Cloudml_Inference_Tie_ModelConfigP33_CD5DC485B54DF4FF6F4BE768FEF8257713_StorageClass
_TtCV15PrivateMLClient43Apple_Cloudml_Inference_Tie_GenerateRequestP33_CD5DC485B54DF4FF6F4BE768FEF8257713_StorageClass
_TtCVVV15PrivateMLClient44Apple_Cloudml_Inference_Tie_GenerateResponse13FinalResponse9DebugInfoP33_CD5DC485B54DF4FF6F4BE768FEF8257713_StorageClass
```

Note: ObjC method lists are effectively empty in this image; this target appears primarily Swift.

Key selectors sample:
```text
0x1fadf98ec: openURL:configuration:completionHandler:
```

## Runtime Executables
### mlruntimed
Summary:
```text
## mlruntimed (x86_64)
file=research/ipsw_ane_adjacent/raw/mlruntimed.x86_64.classdump.txt
lines=211 interfaces=6 protocols=4 objc_methods=76

## PrivateMLClientInferenceProviderService (arm64e)
file=research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.arm64e.classdump.txt
lines=8 interfaces=1 protocols=0 objc_methods=0

## PrivateMLClientInferenceProviderService (x86_64)
file=research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.x86_64.classdump.txt
```

Dependencies:
```text
/usr/libexec/mlruntimed:
	/System/Library/PrivateFrameworks/Trial.framework/Versions/A/Trial (compatibility version 1.0.0, current version 474.2.0, weak)
	/System/Library/PrivateFrameworks/CoreAnalytics.framework/Versions/A/CoreAnalytics (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/PrivateFrameworks/DistributedEvaluation.framework/Versions/A/DistributedEvaluation (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation (compatibility version 300.0.0, current version 4302.0.0)
	/System/Library/PrivateFrameworks/MLRuntime.framework/Versions/A/MLRuntime (compatibility version 1.0.0, current version 1.0.0)
	/usr/lib/libobjc.A.dylib (compatibility version 1.0.0, current version 228.0.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)
	/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 4302.0.0)
	/System/Library/PrivateFrameworks/MobileKeyBag.framework/Versions/A/MobileKeyBag (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/PrivateFrameworks/TrialProto.framework/Versions/A/TrialProto (compatibility version 1.0.0, current version 474.2.0)
```

Entitlements (excerpt):
```text
[Dict]
	[Key] application-identifier
	[Value]
		[String] com.apple.mlruntimed
	[Key] com.apple.coreduetd.allow
	[Value]
		[Bool] true
	[Key] com.apple.duet.activityscheduler.allow
	[Value]
		[Bool] true
	[Key] com.apple.photos.bourgeoisie
	[Value]
		[Bool] true
	[Key] com.apple.private.biome.read-write
	[Value]
		[Array]
			[String] Lighthouse.Ledger.MlruntimedEvent
			[String] Lighthouse.Ledger.LighthousePluginEvent
			[String] Lighthouse.Ledger.TrialdEvent
	[Key] com.apple.private.des-service
	[Value]
		[Bool] true
	[Key] com.apple.private.dprivacyd.allow
	[Value]
		[Bool] true
	[Key] com.apple.private.dprivacyd.metadata.allow
	[Value]
		[Bool] true
	[Key] com.apple.private.photoanalysisd.access
	[Value]
		[Bool] true
	[Key] com.apple.private.speech-model-training
	[Value]
		[Bool] true
	[Key] com.apple.private.xpc.domain-extension
	[Value]
		[Bool] true
	[Key] com.apple.proactive.PersonalizationPortrait.Contact
	[Value]
		[Bool] true
	[Key] com.apple.proactive.PersonalizationPortrait.NamedEntity.readOnly
	[Value]
		[Bool] true
	[Key] com.apple.siriinferenced
	[Value]
		[Bool] true
	[Key] com.apple.trial.client
	[Value]
		[Array]
			[String] 1580
	[Key] com.apple.trial.status.evaluation.mlruntime.allow
	[Value]
		[Bool] true
```

### PrivateMLClientInferenceProviderService
Summary:
```text
lines=8 interfaces=1 protocols=0 objc_methods=0

```

Dependencies:
```text
/System/Library/ExtensionKit/Extensions/PrivateMLClientInferenceProviderService.appex/Contents/MacOS/PrivateMLClientInferenceProviderService:
	/System/Library/PrivateFrameworks/PrivateMLClientInferenceProvider.framework/Versions/A/PrivateMLClientInferenceProvider (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/PrivateFrameworks/ModelManagerServices.framework/Versions/A/ModelManagerServices (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation (compatibility version 300.0.0, current version 4302.0.0)
	/usr/lib/libobjc.A.dylib (compatibility version 1.0.0, current version 228.0.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)
	/System/Library/Frameworks/ExtensionFoundation.framework/Versions/A/ExtensionFoundation (compatibility version 1.0.0, current version 97.0.0)
	/usr/lib/swift/libswiftAccelerate.dylib (compatibility version 1.0.0, current version 77.0.0, weak)
	/usr/lib/swift/libswiftCore.dylib (compatibility version 0.0.0, current version 0.0.0)
	/usr/lib/swift/libswiftCoreFoundation.dylib (compatibility version 1.0.0, current version 120.100.0, weak)
	/usr/lib/swift/libswiftCoreImage.dylib (compatibility version 1.0.0, current version 2.2.0, weak)
	/usr/lib/swift/libswiftDispatch.dylib (compatibility version 1.0.0, current version 56.0.0, weak)
	/usr/lib/swift/libswiftIOKit.dylib (compatibility version 1.0.0, current version 1.0.0, weak)
	/usr/lib/swift/libswiftMetal.dylib (compatibility version 1.0.0, current version 371.5.0, weak)
	/usr/lib/swift/libswiftOSLog.dylib (compatibility version 1.0.0, current version 10.0.0, weak)
	/usr/lib/swift/libswiftObjectiveC.dylib (compatibility version 1.0.0, current version 951.1.0, weak)
	/usr/lib/swift/libswiftQuartzCore.dylib (compatibility version 1.0.0, current version 5.0.0, weak)
	/usr/lib/swift/libswiftUniformTypeIdentifiers.dylib (compatibility version 1.0.0, current version 877.3.4, weak)
	/usr/lib/swift/libswiftXPC.dylib (compatibility version 1.0.0, current version 105.0.14, weak)
	/usr/lib/swift/libswift_Builtin_float.dylib (compatibility version 1.0.0, current version 0.0.0, weak)
	/usr/lib/swift/libswift_Concurrency.dylib (compatibility version 1.0.0, current version 0.0.0)
	/usr/lib/swift/libswiftos.dylib (compatibility version 1.0.0, current version 1076.40.2)
	/usr/lib/swift/libswiftsimd.dylib (compatibility version 1.0.0, current version 23.0.0, weak)
```

Entitlements (excerpt):
```text
[Dict]
	[Key] application-identifier
	[Value]
		[String] com.apple.privatemlclient.inferenceproviderservice
	[Key] com.apple.PerfPowerServices.data-donation
	[Value]
		[Bool] true
	[Key] com.apple.modelcatalog.full-access
	[Value]
		[Bool] true
	[Key] com.apple.private.assets.accessible-asset-types
	[Value]
		[Array]
			[String] com.apple.MobileAsset.UAF.FM.GenerativeModels
			[String] 
	[Key] com.apple.private.biome.client-identifier
	[Value]
		[String] com.apple.privatemlclient.inferenceproviderservice
	[Key] com.apple.private.biome.read-write
	[Value]
		[Array]
			[String] GenerativeModels.GenerativeFunctions.Instrumentation
			[String] GenerativeExperiences.TransparencyLog
	[Key] com.apple.private.intelligenceplatform.use-cases
	[Value]
		[Dict]
			[Key] com.apple.AppleIntelligence.Reporting.Invocation.Step
			[Value]
				[Dict]
					[Key] Streams
					[Value]
						[Dict]
							[Key] AppleIntelligence.Reporting.Invocation.Step
							[Value]
								[Dict]
									[Key] mode
									[Value]
										[String] read-write
	[Key] com.apple.private.xpc.launchd.per-user-lookup
	[Value]
		[Bool] true
	[Key] com.apple.privatecloudcompute.bundleIdentifierOverride
	[Value]
		[Bool] true
	[Key] com.apple.privatecloudcompute.prefetchRequest
	[Value]
		[Bool] true
	[Key] com.apple.privatecloudcompute.requests
	[Value]
		[Bool] true
	[Key] com.apple.security.attestation.access
	[Value]
		[Bool] true
	[Key] com.apple.security.exception.files.absolute-path.read-only
	[Value]
		[Array]
			[String] /private/var/MobileAsset/AssetsV2/locks/com.apple.UnifiedAssetFramework/
			[String] /private/var/MobileAsset/AssetsV2/com_apple_MobileAsset_UAF_FM_GenerativeModels/purpose_auto/
			[String] /private/var/mobile/Library/com.apple.modelcatalog/sideload/
			[String] /private/var/MobileAsset/PreinstalledAssetsV2/InstallWithOs/com_apple_MobileAsset_UAF_FM_GenerativeModels/
			[String] /System/Library/PreinstalledAssetsV2/RequiredByOs/com_apple_MobileAsset_UAF_FM_GenerativeModels/
	[Key] com.apple.security.exception.mach-lookup.global-name
	[Value]
		[Array]
			[String] com.apple.privatecloudcompute
			[String] com.apple.modelcatalog.catalog
			[String] com.apple.PerfPowerTelemetryClientRegistrationService
			[String] com.apple.mobileasset.autoasset
			[String] com.apple.mobileassetd.v2
	[Key] com.apple.security.exception.shared-preference.read-only
	[Value]
		[Array]
			[String] com.apple.tokengeneration
			[String] com.apple.privatecloudcompute
			[String] com.apple.modelcatalog.multiBaseModel
	[Key] com.apple.security.exception.shared-preference.read-write
	[Value]
		[Array]
			[String] com.apple.privatemlclient
	[Key] com.apple.security.temporary-exception.files.absolute-path.read-only
	[Value]
		[Array]
			[String] /Library/Preferences/com.apple.tokengeneration.plist
			[String] /Library/Preferences/com.apple.privatemlclient.plist
	[Key] com.apple.security.temporary-exception.mach-lookup.global-name
	[Value]
		[Array]
			[String] com.apple.privatecloudcompute
			[String] com.apple.modelcatalog.catalog
			[String] com.apple.PerfPowerTelemetryClientRegistrationService
	[Key] com.apple.security.temporary-exception.shared-preference.read-only
	[Value]
		[Array]
			[String] com.apple.tokengeneration
			[String] com.apple.privatecloudcompute
	[Key] com.apple.security.temporary-exception.shared-preference.read-write
	[Value]
		[Array]
			[String] com.apple.privatemlclient
	[Key] com.apple.springboard.opensensitiveurl
	[Value]
		[Bool] true
```

## Priority Notes
- High yield confirmed: Espresso (largest ObjC surface), RemoteCoreML (compact but concrete networking/inference control surface).
- Medium-high confirmed: MLRuntime (task/session/plugin orchestration).
- Medium confirmed: PrivateMLClient (mostly Swift metadata in ObjC dump; better candidate for Swift-focused analysis next).

## Raw Artifacts
```text
research/ipsw_ane_adjacent/raw/Espresso.dsc.classdump.re.txt
research/ipsw_ane_adjacent/raw/Espresso.dsc.classdump.refs.txt
research/ipsw_ane_adjacent/raw/Espresso.dsc.classdump.txt
research/ipsw_ane_adjacent/raw/Espresso.dyld.macho.objc.err
research/ipsw_ane_adjacent/raw/Espresso.dyld.macho.objc.txt
research/ipsw_ane_adjacent/raw/Espresso.dyld.macho.objc_refs.err
research/ipsw_ane_adjacent/raw/Espresso.dyld.macho.objc_refs.txt
research/ipsw_ane_adjacent/raw/Espresso.dyld.objc.class.txt
research/ipsw_ane_adjacent/raw/Espresso.dyld.objc.proto.txt
research/ipsw_ane_adjacent/raw/Espresso.dyld.objc.sel.txt
research/ipsw_ane_adjacent/raw/MLRuntime.dsc.classdump.re.txt
research/ipsw_ane_adjacent/raw/MLRuntime.dsc.classdump.refs.txt
research/ipsw_ane_adjacent/raw/MLRuntime.dsc.classdump.txt
research/ipsw_ane_adjacent/raw/MLRuntime.dyld.macho.objc.err
research/ipsw_ane_adjacent/raw/MLRuntime.dyld.macho.objc.txt
research/ipsw_ane_adjacent/raw/MLRuntime.dyld.macho.objc_refs.err
research/ipsw_ane_adjacent/raw/MLRuntime.dyld.macho.objc_refs.txt
research/ipsw_ane_adjacent/raw/MLRuntime.dyld.objc.class.txt
research/ipsw_ane_adjacent/raw/MLRuntime.dyld.objc.proto.txt
research/ipsw_ane_adjacent/raw/MLRuntime.dyld.objc.sel.txt
research/ipsw_ane_adjacent/raw/PrivateMLClient.dsc.classdump.re.txt
research/ipsw_ane_adjacent/raw/PrivateMLClient.dsc.classdump.refs.txt
research/ipsw_ane_adjacent/raw/PrivateMLClient.dsc.classdump.txt
research/ipsw_ane_adjacent/raw/PrivateMLClient.dyld.macho.objc.err
research/ipsw_ane_adjacent/raw/PrivateMLClient.dyld.macho.objc.txt
research/ipsw_ane_adjacent/raw/PrivateMLClient.dyld.macho.objc_refs.err
research/ipsw_ane_adjacent/raw/PrivateMLClient.dyld.macho.objc_refs.txt
research/ipsw_ane_adjacent/raw/PrivateMLClient.dyld.objc.class.txt
research/ipsw_ane_adjacent/raw/PrivateMLClient.dyld.objc.proto.txt
research/ipsw_ane_adjacent/raw/PrivateMLClient.dyld.objc.sel.txt
research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.arm64e.classdump.err
research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.arm64e.classdump.txt
research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.arm64e.macho.objc.err
research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.arm64e.macho.objc.txt
research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.arm64e.macho.objc_refs.err
research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.arm64e.macho.objc_refs.txt
research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.deps.txt
research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.entitlements.xml
research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.x86_64.classdump.err
research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.x86_64.classdump.txt
research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.x86_64.macho.objc.err
research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.x86_64.macho.objc.txt
research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.x86_64.macho.objc_refs.err
research/ipsw_ane_adjacent/raw/PrivateMLClientInferenceProviderService.x86_64.macho.objc_refs.txt
research/ipsw_ane_adjacent/raw/RemoteCoreML.dsc.classdump.re.txt
research/ipsw_ane_adjacent/raw/RemoteCoreML.dsc.classdump.refs.txt
research/ipsw_ane_adjacent/raw/RemoteCoreML.dsc.classdump.txt
research/ipsw_ane_adjacent/raw/RemoteCoreML.dyld.macho.objc.err
research/ipsw_ane_adjacent/raw/RemoteCoreML.dyld.macho.objc.txt
research/ipsw_ane_adjacent/raw/RemoteCoreML.dyld.macho.objc_refs.err
research/ipsw_ane_adjacent/raw/RemoteCoreML.dyld.macho.objc_refs.txt
research/ipsw_ane_adjacent/raw/RemoteCoreML.dyld.objc.class.txt
research/ipsw_ane_adjacent/raw/RemoteCoreML.dyld.objc.proto.txt
research/ipsw_ane_adjacent/raw/RemoteCoreML.dyld.objc.sel.txt
research/ipsw_ane_adjacent/raw/mlruntimed.arm64e.classdump.err
research/ipsw_ane_adjacent/raw/mlruntimed.arm64e.classdump.txt
research/ipsw_ane_adjacent/raw/mlruntimed.arm64e.macho.objc.err
research/ipsw_ane_adjacent/raw/mlruntimed.arm64e.macho.objc.txt
research/ipsw_ane_adjacent/raw/mlruntimed.arm64e.macho.objc_refs.err
research/ipsw_ane_adjacent/raw/mlruntimed.arm64e.macho.objc_refs.txt
research/ipsw_ane_adjacent/raw/mlruntimed.deps.txt
research/ipsw_ane_adjacent/raw/mlruntimed.entitlements.xml
research/ipsw_ane_adjacent/raw/mlruntimed.x86_64.classdump.err
research/ipsw_ane_adjacent/raw/mlruntimed.x86_64.classdump.txt
research/ipsw_ane_adjacent/raw/mlruntimed.x86_64.macho.objc.err
research/ipsw_ane_adjacent/raw/mlruntimed.x86_64.macho.objc.txt
research/ipsw_ane_adjacent/raw/mlruntimed.x86_64.macho.objc_refs.err
research/ipsw_ane_adjacent/raw/mlruntimed.x86_64.macho.objc_refs.txt
research/ipsw_ane_adjacent/raw/summary_stats.txt
```
