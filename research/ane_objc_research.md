# ANE Objective-C Recon (ipsw)

Generated: 2026-03-03 22:11:05Z

## Environment
- Host:
```text
=== sw_vers ===
ProductName:		macOS
ProductVersion:		26.3
BuildVersion:		25D125

=== ipsw version ===
Version: , BuildCommit: 
```

## Scope
Used `ipsw` to extract ObjC metadata and selectors from:
- dyld cache images: AppleNeuralEngine, ParavirtualizedANE, ANEServices, ANECompiler, ANEClientSignals, MilAneflow
- XPC binaries: ANECompilerService, ANEStorageMaintainer
- daemons: aned, aneuserd

## Coverage Summary
```text
# Summary stats (generated)
## AppleNeuralEngine
file=research/ipsw_ane/raw/AppleNeuralEngine.dsc.classdump.txt
lines=1376 interfaces=35 protocols=5 objc_methods=734

## ANECompilerService
file=research/ipsw_ane/raw/ANECompilerService.classdump.txt
lines=321 interfaces=15 protocols=5 objc_methods=138

## ANEStorageMaintainer
file=research/ipsw_ane/raw/ANEStorageMaintainer.classdump.txt
lines=107 interfaces=3 protocols=3 objc_methods=45

## ParavirtualizedANE
file=research/ipsw_ane/raw/ParavirtualizedANE.dsc.classdump.txt
lines=175 interfaces=2 protocols=0 objc_methods=129

## ANEServices
file=research/ipsw_ane/raw/ANEServices.dsc.classdump.txt
lines=10 interfaces=1 protocols=0 objc_methods=4

## dyld objc selectors
AppleNeuralEngine 715
ParavirtualizedANE 274
ANEServices 5
```

## AppleNeuralEngine (dyld cache)
Source: `research/ipsw_ane/raw/AppleNeuralEngine.dsc.classdump.txt`

### Classes (35)
```text
NSFileManager
_ANEBuffer
_ANEChainingRequest
_ANEClient
_ANECloneHelper
_ANEDaemonConnection
_ANEDataReporter
_ANEDeviceController
_ANEDeviceInfo
_ANEErrors
_ANEHashEncoding
_ANEIOSurfaceObject
_ANEIOSurfaceOutputSets
_ANEInMemoryModel
_ANEInMemoryModelDescriptor
_ANEInputBuffersReady
_ANELog
_ANEModel
_ANEModelInstanceParameters
_ANEModelToken
_ANEOutputSetEnqueue
_ANEPerformanceStats
_ANEPerformanceStatsIOSurface
_ANEProcedureData
_ANEProgramForEvaluation
_ANEProgramIOSurfacesMapper
_ANEQoSMapper
_ANERequest
_ANESandboxingHelper
_ANESharedEvents
_ANESharedSignalEvent
_ANESharedWaitEvent
_ANEStrings
_ANEVirtualClient
_ANEWeight
```

### Protocols
```text
NSCoding
NSCopying
NSSecureCoding
_ANEDaemonProtocol
_ANEDaemonProtocol_Private
```

### Method-Dense Classes
```text
_ANEVirtualClient	87
_ANEStrings	86
_ANEModel	63
_ANEClient	49
_ANEInMemoryModel	43
_ANEErrors	34
_ANERequest	26
_ANEDaemonConnection	21
_ANEProgramForEvaluation	21
_ANEWeight	18
_ANEDeviceInfo	17
_ANEChainingRequest	16
_ANEDeviceController	16
_ANEInMemoryModelDescriptor	16
_ANEPerformanceStats	16
_ANESharedSignalEvent	16
_ANEIOSurfaceObject	15
_ANEProgramIOSurfacesMapper	14
_ANEModelToken	12
_ANEQoSMapper	12
```

### Core XPC/Daemon Contract
```text
@protocol _ANEDaemonProtocol

@required

/* required instance methods */
-[_ANEDaemonProtocol compileModel:sandboxExtension:options:qos:withReply:];
-[_ANEDaemonProtocol compiledModelExistsFor:withReply:];
-[_ANEDaemonProtocol compiledModelExistsMatchingHash:withReply:];
-[_ANEDaemonProtocol loadModel:sandboxExtension:options:qos:withReply:];
-[_ANEDaemonProtocol loadModelNewInstance:options:modelInstParams:qos:withReply:];
-[_ANEDaemonProtocol prepareChainingWithModel:options:chainingReq:qos:withReply:];
-[_ANEDaemonProtocol purgeCompiledModel:withReply:];
-[_ANEDaemonProtocol purgeCompiledModelMatchingHash:withReply:];
-[_ANEDaemonProtocol reportTelemetryToPPS:playload:];
-[_ANEDaemonProtocol unloadModel:options:qos:withReply:];

@optional

@end

@protocol _ANEDaemonProtocol_Private <_ANEDaemonProtocol>

@required

/* required instance methods */
-[_ANEDaemonProtocol_Private echo:withReply:];
-[_ANEDaemonProtocol_Private beginRealTimeTaskWithReply:];
-[_ANEDaemonProtocol_Private endRealTimeTaskWithReply:];

@optional

@end
```

### Key Selectors (from dyld objc sel)
Source: `research/ipsw_ane/raw/AppleNeuralEngine.dyld.objc.sel.key.txt`
```text
0x1f9e96b90: loadModelNewInstance:options:modelInstParams:qos:error:
0x1fabdf8d4: purgeCompiledModel:
0x1fb3b797e: modelAtURL:key:mpsConstants:
0x1fb01a3f9: modelAttributes
0x1fb3b6589: buffersReadyWithModel:inputBuffers:options:qos:error:
0x1fb3b6a8a: defaultANECIRFileName
0x1fa790390: getCFDictionaryFromIOSurface:dataLength:
0x1fb3b6ff4: initWithCsIdentity:teamIdentity:modelIdentifier:processIdentifier:
0x1fa4be4b0: endRealTimeTask
0x1fa9aa290: numANECores
0x1fb3b8a56: tokenWithCsIdentity:teamIdentity:modelIdentifier:processIdentifier:
0x1fb3b8581: setWeightURL:
0x1fb3b85cf: signalEvents
0x1fb3b7a1f: modelAtURLWithSourceURL:sourceURL:key:identifierSource:cacheURLIdentifier:UUID:
0x1fb3b67eb: compilerOptionsFileName
0x1fb3b8d60: waitEvents
0x1fb3b88b1: testing_external_precompiledModelPath
0x1fb3b6c68: doPrepareChainingWithModel:options:chainingReq:qos:error:
0x1f9df3588: hasANE
0x1fb024b80: objectWithIOSurface:
0x1fa364990: copyDictionaryToIOSurface:copiedDataSize:createdIOSID:
0x1fb3b7dd5: prepareANEMemoryMappingParams:request:
0x1fb3b6aa0: defaultANECIROptionsFileName
0x1fb3472a3: evaluateRealTimeWithModel:options:request:error:
0x1fa0a42bf: compileModel:options:qos:error:
0x1fa05e010: freeModelFileIOSurfaces:
0x1f9dfd810: modelIdentifier
0x1f9e19d77: compiledModelExistsFor:
0x1fb3b8aca: unloadModel:options:qos:withReply:
0x1fa535cdc: copyToIOSurface:length:ioSID:
0x1fb3b8c38: virtualClient
0x1fb3b6c0f: doLoadModel:options:qos:error:
0x1f9e8f0f7: outputDictIOSurfaceSize
0x1fb3b660d: cacheURLIdentifier
0x1fb3b8c5f: virtualizationHostError:
0x1fb3b7d1c: ppsCategoryForANE
0x1fb3b7b10: modelSourceStoreName
0x1fa4ac4b6: beginRealTimeTask
0x1fa677f90: getDictionaryWithJSONEncodingFromIOSurface:withArchivedDataSize:
0x1fb3b87d6: testing_cacheDirectoryWithSuffix:
0x1fb3b74ca: initWithSignalEvents:waitEvents:
0x1fb3b8b15: unmapIOSurfacesWithRequest:
0x1fb3493bc: sharedEventsWithSignalEvents:waitEvents:
0x1f9fde410: doMapIOSurfacesWithModel:request:cacheInference:error:
0x1fb348240: objectWithIOSurface:statType:
0x1fb3b6f6d: initWithANEPrivilegedVM:
0x1fb349409: signalEventWithValue:symbolIndex:eventType:sharedEvent:
0x1fb3b8d36: waitEventWithValue:sharedEvent:eventType:
0x1fb3b8a1f: tokenWithAuditToken:modelIdentifier:processIdentifier:
0x1fb3b7c6c: outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:
0x1fb346ac8: aneRealTimeTaskQoS
0x1fb3b7086: initWithIOSurface:startOffset:shouldRetain:
0x1fb3b6417: adapterWeightsAccessEntitlementBypassBootArg
0x1fa5dfaa7: loadModel:options:qos:error:
0x1fa11d4bd: printIOSurfaceDataInBytes:
0x1fb3b6d05: endRealTimeTaskWithReply:
0x1fb0245d6: modelWithNetworkDescription:weights:optionsPlist:
0x1fb3b6f1a: initOutputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:
0x1f9ff6dba: mapIOSurfacesWithModel:request:cacheInference:error:
0x1fb3b7e34: prepareChainingWithModel:options:chainingReq:qos:withReply:
0x1fb3b79d4: modelAtURLWithSourceURL:sourceURL:key:identifierSource:cacheURLIdentifier:
0x1fa3ef290: model
0x1fa7079c0: purgeCompiledModelMatchingHash:
0x1fb3b7275: initWithModelAtURL:sourceURL:UUID:key:identifierSource:cacheURLIdentifier:modelAttributes:standardizeURL:string_id:generateNewStringId:
0x1fb3b7836: loadModelNewInstance:options:modelInstParams:qos:withReply:
0x1fb3b8278: reportClient:modelName:
0x1fb3b72fd: initWithModelAtURL:sourceURL:UUID:key:identifierSource:cacheURLIdentifier:modelAttributes:standardizeURL:string_id:generateNewStringId:mpsConstants:
0x1fb3b891d: testing_modelNames
0x1fb3b7b47: modelWithMILText:weights:optionsPlist:
0x1fb3b6bda: doEvaluateDirectWithModel:options:request:qos:error:
0x1fb3b7587: initWithWeightSymbolAndURLSHA:weightURL:SHACode:sandboxExtension:
0x1fa3ffd90: releaseIOSurfaces:
0x1fa08042d: doEvaluateWithModelLegacy:options:request:qos:completionEvent:error:
0x1fb3b8c78: virtualizationHostError:error:
0x1fb348528: purgeCompiledModel
0x1fb3b6b6e: doBuffersReadyWithModel:inputBuffers:options:qos:error:
0x1fb3b85ee: signalValue
0x1f9e1ba90: compiler
0x1fb3b6c2e: doLoadModelNewInstance:options:modelInstParams:qos:error:
0x1fb3b7dfc: prepareChainingWithModel:options:chainingReq:qos:error:
0x1fad57923: copyLLIRBundleToIOSurface:writtenDataSize:
0x1fb3b7b25: modelWithCacheURLIdentifier:UUID:
0x1fb3b7969: modelAssetsCacheName
0x1fb3b7f08: processOutputSet:model:options:error:
0x1fb3b81dc: purgeCompiledModel:withReply:
0x1fa206d61: updatePerformanceStats:performanceStatsLength:perfStatsRawIOSurfaceRef:performanceStatsRawLength:hwExecutionTime:
0x1fb3b8c46: virtualizationDataError:
0x1fb3b69b2: createIOSurfaceWithWidth:pixel_size:height:
0x1fb3b6377: adapterWeightsAccessEntitlement
0x1fb3b8224: qosForProgramPriority:
0x1fb3b70b2: initWithIOSurface:statType:
0x1fb3b800e: programIOSurfacesUnmapErrorForMethod:code:
0x1fb3467e3: waitEventWithValue:sharedEvent:
0x1fb3b7ea1: processInputBuffers:model:options:error:
0x1fb3b6803: compilerOptionsWithOptions:isCompiledModelCached:
0x1fae0c798: signaledValue
0x1fb3b6ca2: doUnloadModel:options:qos:error:
0x1f9fb5ead: copyToIOSurface:size:ioSID:
0x1f9eb659e: evaluateWithModel:options:request:qos:error:
0x1fb3b8b9d: updateWeightURL:
0x1fb3b69de: createIOSurfaceWithWidth:pixel_size:height:bytesPerElement:
0x1fa88ca9c: copyFilesInDirectoryToIOSurfaces:ioSurfaceRefs:ioSurfaceSizes:fileNames:
0x1fb3b62f7: modelBinaryName
0x1fb3b8905: testing_modelDirectory:
0x1fb3b8896: testing_external_modelPath
0x1fb3b676c: compileWithQoS:options:error:
0x1fb3b8c21: validateRequest:model:
0x1fa8dd310: createIOSurface:ioSID:
0x1fb3b7bb6: objectWithIOSurface:startOffset:
0x1fa067cb3: readWeightFilename:
0x1fb3b78cb: mapIOSurfacesWithRequest:cacheInference:error:
0x1fb3b62d7: modelAtURL:key:modelAttributes:
0x1fb3b87bf: testing_cacheDirectory
0x1fb3b7173: initWithInputs:outputs:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:
0x1fb3b83fa: sandboxExtensionPathForModelURL:
0x1fa9243b7: compiledModelExistsMatchingHash:
0x1fb3b87f8: testing_cacheDirectoryWithSuffix:buildVersion:
0x1fb3b6cc3: driverMaskForANEFMask:
0x1f9e4e190: doEvaluateWithModel:options:request:qos:completionEvent:error:
0x1fb3b8cc2: vm_allowPrecompiledBinaryBootArg
```

## ANECompilerService (XPC)
Source: `research/ipsw_ane/raw/ANECompilerService.classdump.txt`

### Classes
```text
NSFileManager
ServiceDelegate
_ANECVAIRCompiler
_ANECompiler
_ANECompilerService
_ANECoreMLModelCompiler
_ANEEspressoIRTranslator
_ANEInMemoryModelCacheManager
_ANEMILCompiler
_ANEMLIRCompiler
_ANEModelCacheManager
_ANESandboxingHelper
_ANEStorageHelper
_ANETask
_ANETaskManager
```

### Protocols
```text
NSObject
NSXPCListenerDelegate
_ANECompilerServiceProtocol
_ANEMaintenanceProtocol
_ANEStorageMaintainerProtocol
```

### Service Protocol
```text
@protocol _ANECompilerServiceProtocol

@required

/* required instance methods */
-[_ANECompilerServiceProtocol compileModelAt:csIdentity:sandboxExtension:options:tempDirectory:cloneDirectory:outputURL:aotModelBinaryPath:withReply:];

@optional

@end
```

## ANEStorageMaintainer (XPC)
Source: `research/ipsw_ane/raw/ANEStorageMaintainer.classdump.txt`

### Classes and Protocol
```text
ServiceDelegate
_ANEStorageHelper
_ANEStorageMaintainer

@protocol _ANEStorageMaintainerProtocol

@required

/* required instance methods */
-[_ANEStorageMaintainerProtocol purgeDanglingModelsAt:withReply:];

@optional

@end
```

## aned / aneuserd (explicit arch dump)
Source: `research/ipsw_ane/raw/aned.arm64e.classdump.txt`, `research/ipsw_ane/raw/aneuserd.arm64e.classdump.txt`

### Classes
```text
_ANEInMemoryModelCacheManager
_ANEModelCacheManager
_ANEProgramCache
_ANEProgramCacheKey
_ANEProgramForLoad
_ANEServer
_ANEStorageHelper
_ANETask
_ANETaskManager
_ANETemporaryFilesHandler
_ANEXPCServiceHelper
```

### _ANEServer
```text
@interface _ANEServer : NSObject <_ANEDaemonProtocol, _ANEDaemonProtocol_Private> {
    /* instance variables */
    B _isRootDaemon;
    @"_ANEModelCacheManager" _modelAssetCacheManager;
    @"_ANEInMemoryModelCacheManager" _modelCacheManager;
    @"NSArray" _semaArray;
    @"_ANEXPCServiceHelper" _unrestricted;
    @"_ANEXPCServiceHelper" _restricted;
    @"_ANEXPCServiceHelper" _unrestrictedUser;
    @"NSUUID" _uuidANECompilerServiceLongerDuration;
    @"NSUUID" _uuidANECompilerServiceRegular;
}

@property (T@"_ANEModelCacheManager",R,N,V_modelAssetCacheManager) modelAssetCacheManager;
@property (T@"_ANEInMemoryModelCacheManager",R,N,V_modelCacheManager) modelCacheManager;
@property (T@"NSArray",R,N,V_semaArray) semaArray;
@property (T@"_ANEXPCServiceHelper",R,N,V_unrestricted) unrestricted;
@property (T@"_ANEXPCServiceHelper",R,N,V_restricted) restricted;
@property (T@"_ANEXPCServiceHelper",R,N,V_unrestrictedUser) unrestrictedUser;
@property (T@"NSUUID",R,N,V_uuidANECompilerServiceLongerDuration) uuidANECompilerServiceLongerDuration;
@property (T@"NSUUID",R,N,V_uuidANECompilerServiceRegular) uuidANECompilerServiceRegular;
@property (TB,R,N,V_isRootDaemon) isRootDaemon;

/* class methods */
+[_ANEServer initialize];

/* instance methods */
-[_ANEServer init];
-[_ANEServer initUser];
-[_ANEServer initWithDataVaultDirectory:dataVaultStorageClass:buildVersion:tempDirectory:cloneDirectory:];
-[_ANEServer start];
-[_ANEServer startUser];
-[_ANEServer handleLaunchServicesEvent:userInfo:];
-[_ANEServer _handleIOKitEvent:];
-[_ANEServer handleIOKitEvent:];
-[_ANEServer beginRealTimeTaskWithReply:];
-[_ANEServer endRealTimeTaskWithReply:];
-[_ANEServer echo:withReply:];
-[_ANEServer reportTelemetryToPPS:playload:];
-[_ANEServer compiledModelExistsFor:withReply:];
-[_ANEServer purgeCompiledModel:withReply:];
-[_ANEServer compileModel:sandboxExtension:options:qos:withReply:];
-[_ANEServer loadModel:sandboxExtension:options:qos:withReply:];
-[_ANEServer loadModelNewInstance:options:modelInstParams:qos:withReply:];
-[_ANEServer unloadModel:options:qos:withReply:];
-[_ANEServer prepareChainingWithModel:options:chainingReq:qos:withReply:];
-[_ANEServer compiledModelExistsMatchingHash:withReply:];
-[_ANEServer purgeCompiledModelMatchingHash:withReply:];
-[_ANEServer compileAsNeededAndLoadCachedModel:csIdentity:sandboxExtension:options:qos:modelFilePath:modelAttributes:error:];
-[_ANEServer doCompileModel:csIdentity:sandboxExtension:options:qos:withReply:];
-[_ANEServer modelAssetCacheManager];
-[_ANEServer modelCacheManager];
-[_ANEServer semaArray];
-[_ANEServer unrestricted];
-[_ANEServer restricted];
-[_ANEServer unrestrictedUser];
-[_ANEServer uuidANECompilerServiceLongerDuration];
-[_ANEServer uuidANECompilerServiceRegular];
-[_ANEServer isRootDaemon];

@end
```

### _ANEXPCServiceHelper
```text
@interface _ANEXPCServiceHelper : NSObject <NSXPCListenerDelegate, NSXPCConnectionDelegate> {
    /* instance variables */
    B _requiresEntitlement;
    @"NSXPCListener" _listener;
    @"NSString" _serviceName;
    @"NSXPCInterface" _interface;
    @"NSString" _entitlementString;
    @ _server;
}

@property (T@"NSString",C,N,V_serviceName) serviceName;
@property (T@"NSXPCInterface",R,N,V_interface) interface;
@property (TB,R,N,V_requiresEntitlement) requiresEntitlement;
@property (T@"NSString",C,N,V_entitlementString) entitlementString;
@property (T@,R,N,V_server) server;
@property (T@"NSXPCListener",R,N,V_listener) listener;
@property (TQ,R) hash;
@property (T#,R) superclass;
@property (T@"NSString",R,C) description;
@property (T@"NSString",?,R,C) debugDescription;

/* class methods */
+[_ANEXPCServiceHelper new];
+[_ANEXPCServiceHelper serviceWithName:interface:delegate:requiresEntitlement:entitlementString:];
+[_ANEXPCServiceHelper allowRestrictedAccessFor:];
+[_ANEXPCServiceHelper allowAggressivePowerSavingFor:];
+[_ANEXPCServiceHelper allowProcessModelShareFor:];
+[_ANEXPCServiceHelper allowRestrictedAccessFor:entitlementString:];

/* instance methods */
-[_ANEXPCServiceHelper init];
-[_ANEXPCServiceHelper initWithMachServiceName:interface:delegate:requiresEntitlement:entitlementString:];
-[_ANEXPCServiceHelper listener:shouldAcceptNewConnection:];
-[_ANEXPCServiceHelper listener];
-[_ANEXPCServiceHelper serviceName];
-[_ANEXPCServiceHelper setServiceName:];
-[_ANEXPCServiceHelper interface];
-[_ANEXPCServiceHelper requiresEntitlement];
-[_ANEXPCServiceHelper entitlementString];
-[_ANEXPCServiceHelper setEntitlementString:];
-[_ANEXPCServiceHelper server];

@end
```

## ParavirtualizedANE (dyld cache)
Source: `research/ipsw_ane/raw/ParavirtualizedANE.dsc.classdump.txt`

### Classes
```text
_ANEVirtualModel
_ANEVirtualPlatformClient
```

### _ANEVirtualPlatformClient (methods excerpt)
```text
@interface _ANEVirtualPlatformClient : NSObject {
    /* instance variables */
    B _avpSaveRestoreSupported;
    I _negotiatedDataInterfaceVersion;
    @"NSMutableDictionary" _aneModelDict;
    @"_ANEClient" _aneClient;
    @"NSString" _anebaseDirectory;
    @"NSObject<OS_dispatch_queue>" _queue;
    @"NSString" _guestBuildVersion;
    Q _negotiatedCapabilityMask;
}

@property (T@"NSMutableDictionary",R,N,V_aneModelDict) aneModelDict;
@property (T@"_ANEClient",R,N,V_aneClient) aneClient;
@property (T@"NSString",R,N,V_anebaseDirectory) anebaseDirectory;
@property (T@"NSObject<OS_dispatch_queue>",R,N,V_queue) queue;
@property (T@"NSString",&,N,V_guestBuildVersion) guestBuildVersion;
@property (TI,N,V_negotiatedDataInterfaceVersion) negotiatedDataInterfaceVersion;
@property (TQ,N,V_negotiatedCapabilityMask) negotiatedCapabilityMask;
@property (TB,N,V_avpSaveRestoreSupported) avpSaveRestoreSupported;

/* class methods */
+[_ANEVirtualPlatformClient new];
+[_ANEVirtualPlatformClient initialize];
+[_ANEVirtualPlatformClient dictionaryGetUInt64ForKey:key:];
+[_ANEVirtualPlatformClient printStruct:];
+[_ANEVirtualPlatformClient dictionaryGetInt8ForKey:key:];
+[_ANEVirtualPlatformClient sharedConnection];
+[_ANEVirtualPlatformClient createLastModificationTimeStampFile:fileName:timeSince1970:];
+[_ANEVirtualPlatformClient dictionaryGetDataBytesForKey:key:];
+[_ANEVirtualPlatformClient generateExpectedBaseDirectoryWith:withCSIdentity:withVMID:fileSubDirectory:];
+[_ANEVirtualPlatformClient generateExpectedPathForFileName:withANEBaseDirectory:withCSIdentity:withVMID:fileSubDirectory:];
+[_ANEVirtualPlatformClient getCodeSigningIdentity:];
+[_ANEVirtualPlatformClient getCommandInputSize:];
+[_ANEVirtualPlatformClient getCommandOutputSize:];
+[_ANEVirtualPlatformClient getDataFromIOSurface:withSize:];
+[_ANEVirtualPlatformClient getPrecompiledModelPath:error:];
+[_ANEVirtualPlatformClient isFileModificationTimeStampSame:timeSince1970:];
+[_ANEVirtualPlatformClient isValidBufferSize:];
+[_ANEVirtualPlatformClient needToSaveFile:fileName:timeSince1970:];
+[_ANEVirtualPlatformClient prepareValidationResultForSerialization:];
+[_ANEVirtualPlatformClient sharedConnection:];

/* instance methods */
-[_ANEVirtualPlatformClient compiledModelExistsFor:];
-[_ANEVirtualPlatformClient negotiatedDataInterfaceVersion];
-[_ANEVirtualPlatformClient printIOSurfaceDataInBytes:];
-[_ANEVirtualPlatformClient init:];
-[_ANEVirtualPlatformClient purgeCompiledModelMatchingHash:];
-[_ANEVirtualPlatformClient echo:];
-[_ANEVirtualPlatformClient negotiatedCapabilityMask];
-[_ANEVirtualPlatformClient printDictionary:];
-[_ANEVirtualPlatformClient dealloc];
-[_ANEVirtualPlatformClient compiledModelExistsMatchingHash:];
-[_ANEVirtualPlatformClient purgeCompiledModel:];
-[_ANEVirtualPlatformClient queue];
-[_ANEVirtualPlatformClient loadModel:];
-[_ANEVirtualPlatformClient getDeviceInfo:];
-[_ANEVirtualPlatformClient generatedExpectedBaseDirectoryWithFileType:withInputDictionary:];
-[_ANEVirtualPlatformClient echoDictionary:];
-[_ANEVirtualPlatformClient generateExpectedModelFilePathFromDictionary:withFileType:error:];
-[_ANEVirtualPlatformClient aneClient];
-[_ANEVirtualPlatformClient aneModelDict];
-[_ANEVirtualPlatformClient anebaseDirectory];
-[_ANEVirtualPlatformClient asyncAvpHoldMessageReceived:];
-[_ANEVirtualPlatformClient avpSaveRestoreSupported];
-[_ANEVirtualPlatformClient beginRealTimeTask:];
-[_ANEVirtualPlatformClient callCFDictionaryMethod:];
-[_ANEVirtualPlatformClient chunkedDataReceived:];
-[_ANEVirtualPlatformClient compileModel:];
-[_ANEVirtualPlatformClient compileModelDictionary:];
-[_ANEVirtualPlatformClient compiledModelExistsForDictionary:];
-[_ANEVirtualPlatformClient compiledModelExistsMatchingHashDictionary:];
-[_ANEVirtualPlatformClient constructANEModel:modelDirectory:];
-[_ANEVirtualPlatformClient copyBytesToIOSurface:dataLength:ioSID:];
-[_ANEVirtualPlatformClient copyDictionaryUsingJSONSerialization:toIOSurfaceWithID:withMaxSize:outWrittentDataSize:];
-[_ANEVirtualPlatformClient createANEModelInstanceParameters:modelUUID:];
-[_ANEVirtualPlatformClient createAllModelFilesOnHost:];
-[_ANEVirtualPlatformClient createLLIRBundleOnDiskFromIOSurfaceWithID:destDirectory:llirDataSize:];
-[_ANEVirtualPlatformClient createModelFile:path:ioSID:length:];
-[_ANEVirtualPlatformClient createModelFileAtPath:fromIOSID:withDataLength:shouldAppend:];
-[_ANEVirtualPlatformClient createOptionFilesForCompilation:aneModelKey:options:];
-[_ANEVirtualPlatformClient createOptionFilesForCompilationDictionary:aneModelKey:options:modelDirectory:];
-[_ANEVirtualPlatformClient doCreateModelFile:aneModelKey:];
-[_ANEVirtualPlatformClient doModelAttributeUpdate:virtualANEModel:];
-[_ANEVirtualPlatformClient doModelAttributeUpdateDictionary:dictionary:];
-[_ANEVirtualPlatformClient doRemoveModelFile:];
-[_ANEVirtualPlatformClient endRealTimeTask:];
-[_ANEVirtualPlatformClient evaluateWithModel:];
-[_ANEVirtualPlatformClient evaluateWithModelDictionary:];
-[_ANEVirtualPlatformClient exchangeBuildVersionInfo:];
-[_ANEVirtualPlatformClient generateExpectedModelFilePathFromDictionary:error:];
-[_ANEVirtualPlatformClient generateExpectedPathForFileName:withFileType:withInputDictionary:];
-[_ANEVirtualPlatformClient getANEModelKey:caller:];
-[_ANEVirtualPlatformClient getArrayFromIOSurfaceID:localData:localDataLength:];
-[_ANEVirtualPlatformClient getBytesFromIOSurfaceID:dest:dataLength:];
-[_ANEVirtualPlatformClient getDeviceInfoDictionary:];
-[_ANEVirtualPlatformClient getIOSurface:];
-[_ANEVirtualPlatformClient getObjectFromIOSurfaceID:classType:length:];
-[_ANEVirtualPlatformClient getObjectFromIOSurfaceID:classes:topLevelClass:length:];
-[_ANEVirtualPlatformClient getOptionsDictFromInputDict:];
-[_ANEVirtualPlatformClient getQueueCount];
-[_ANEVirtualPlatformClient guestBuildVersion];
-[_ANEVirtualPlatformClient loadModelDictionary:];
-[_ANEVirtualPlatformClient loadModelNewInstance:];
-[_ANEVirtualPlatformClient loadModelNewInstanceLegacy:];
-[_ANEVirtualPlatformClient mapIOSurfacesWithModel:];
-[_ANEVirtualPlatformClient mapIOSurfacesWithModelDictionary:ioSIDs:ioSIDsSize:];
-[_ANEVirtualPlatformClient modelAction:buffer:bufferSize:];
-[_ANEVirtualPlatformClient objectWithIOSurfaceID:];
-[_ANEVirtualPlatformClient purgeCompiledModelDictionary:];
-[_ANEVirtualPlatformClient purgeCompiledModelMatchingHashDictionary:];
-[_ANEVirtualPlatformClient removeAllModelFiles:];
-[_ANEVirtualPlatformClient requestWithVirtualANEModel:];
-[_ANEVirtualPlatformClient requestWithVirtualANEModelDictionary:];
-[_ANEVirtualPlatformClient sessionHintWithModel:];
-[_ANEVirtualPlatformClient setAvpSaveRestoreSupported:];
-[_ANEVirtualPlatformClient setGuestBuildVersion:];
-[_ANEVirtualPlatformClient setNegotiatedCapabilityMask:];
-[_ANEVirtualPlatformClient setNegotiatedDataInterfaceVersion:];
-[_ANEVirtualPlatformClient storeGuestBuildVersion:];
-[_ANEVirtualPlatformClient unLoadModel:];
-[_ANEVirtualPlatformClient unLoadModelDictionary:];
-[_ANEVirtualPlatformClient updateErrorValue:error:];
-[_ANEVirtualPlatformClient updateErrorValueDictionary:error:];
-[_ANEVirtualPlatformClient updateErrorValueForIOSID:error:dictToUpdate:maxIOSurfaceSize:];
-[_ANEVirtualPlatformClient updateOptions:virtualANEModel:];
-[_ANEVirtualPlatformClient updateOptionsDictionary:directoryPath:];
-[_ANEVirtualPlatformClient updatePerformanceStats:aneRequest:];
-[_ANEVirtualPlatformClient updatePerformanceStatsDictionary:aneRequest:];
-[_ANEVirtualPlatformClient updateVirtualANEModel:model:];
-[_ANEVirtualPlatformClient updateVirtualANEModelDictionary:model:cacheURLIdentifierToGuest:];
-[_ANEVirtualPlatformClient validateNetworkCreate:];
-[_ANEVirtualPlatformClient validateNetworkCreateMLIR:];
-[_ANEVirtualPlatformClient writeDictionaryToIOSurfaceID:dict:writtenDataLength:ioSurfaceMaxSize:];
-[_ANEVirtualPlatformClient writeObjectToIOSurfaceID:obj:writtenDataLength:ioSurfaceMaxSize:];

@end
```

### Key Selectors (from dyld objc sel)
Source: `research/ipsw_ane/raw/ParavirtualizedANE.dyld.objc.sel.key.txt`
```text
0x1fc950043: loadModelNewInstance:
0x1fc9504c6: updateVirtualANEModelDictionary:model:cacheURLIdentifierToGuest:
0x1fb3b8b9d: updateWeightURL:
0x1f9e19d77: compiledModelExistsFor:
0x1fc94feec: getIOSurface:
0x1fb3b7b25: modelWithCacheURLIdentifier:UUID:
0x1fc9503f9: updateOptions:virtualANEModel:
0x1fc94feb4: getDataFromIOSurface:withSize:
0x1fa3ef290: model
0x1fabdf8d4: purgeCompiledModel:
0x1fc94fd5d: generateExpectedPathForFileName:withANEBaseDirectory:withCSIdentity:withVMID:fileSubDirectory:
0x1fc94fe48: getBytesFromIOSurfaceID:dest:dataLength:
0x1fa9243b7: compiledModelExistsMatchingHash:
0x1fa0a42bf: compileModel:options:qos:error:
0x1fc9503bb: updateErrorValueForIOSID:error:dictToUpdate:maxIOSurfaceSize:
0x1fc95002e: loadModelDictionary:
0x1f9dfb990: modelURL
0x1fc9502f7: setTmpWeightFilesPath:
0x1fc94f91a: compileModel:
0x1fc94f9af: copyBytesToIOSurface:dataLength:ioSID:
0x1fc94fc1b: doModelAttributeUpdate:virtualANEModel:
0x1fc94ff79: getPrecompiledModelPath:error:
0x1fa5dfaa7: loadModel:options:qos:error:
0x1f9e96b90: loadModelNewInstance:options:modelInstParams:qos:error:
0x1fb01a3f9: modelAttributes
0x1fb3b660d: cacheURLIdentifier
0x1fc95035d: unLoadModel:
0x1fc950226: requestWithVirtualANEModelDictionary:
0x1fc95036a: unLoadModelDictionary:
0x1fa7079c0: purgeCompiledModelMatchingHash:
0x1fc950539: writeDictionaryToIOSurfaceID:dict:writtenDataLength:ioSurfaceMaxSize:
0x1fc94fb8c: createOptionFilesForCompilationDictionary:aneModelKey:options:modelDirectory:
0x1f9ea8bba: unloadModel:options:qos:error:
0x1fc950059: loadModelNewInstanceLegacy:
0x1fa762190: numANEs
0x1fc94ffb8: initWithModel:tmpModelFilesPath:tmpWeightFilesPath:
0x1fc94f940: compiledModelExistsForDictionary:
0x1f9df3588: hasANE
0x1fc95034a: tmpWeightFilesPath
0x1fb3b7bd7: objectWithIOSurfaceNoRetain:startOffset:
0x1fc94fe15: getArrayFromIOSurfaceID:localData:localDataLength:
0x1faf51ab6: loadModel:
0x1fa9aa290: numANECores
0x1fc9501ac: purgeCompiledModelMatchingHashDictionary:
0x1fb3b8c5f: virtualizationHostError:
0x1fc94fdfe: getANEModelKey:caller:
0x1fc94fefa: getObjectFromIOSurfaceID:classType:length:
0x1f9ff6dba: mapIOSurfacesWithModel:request:cacheInference:error:
0x1fc95012f: objectWithModel:tmpModelFilesPath:tmpWeightFilesPath:
0x1fc95008d: mapIOSurfacesWithModelDictionary:ioSIDs:ioSIDsSize:
0x1f9eb6f99: buildVersion
0x1fc94f962: compiledModelExistsMatchingHashDictionary:
0x1fc94ff25: getObjectFromIOSurfaceID:classes:topLevelClass:length:
0x1fc950107: objectWithIOSurfaceID:
0x1fc9500c1: modelAction:buffer:bufferSize:
0x1fc94f928: compileModelDictionary:
0x1fc94fa2e: createANEModelInstanceParameters:modelUUID:
0x1fb3b7a1f: modelAtURLWithSourceURL:sourceURL:key:identifierSource:cacheURLIdentifier:UUID:
0x1fc95057f: writeObjectToIOSurfaceID:obj:writtenDataLength:ioSurfaceMaxSize:
0x1fc94fa75: createLLIRBundleOnDiskFromIOSurfaceWithID:destDirectory:llirDataSize:
0x1fc95020a: requestWithVirtualANEModel:
0x1fc950075: mapIOSurfacesWithModel:
0x1fc9504a9: updateVirtualANEModel:model:
0x1fc94f9d6: copyDictionaryUsingJSONSerialization:toIOSurfaceWithID:withMaxSize:outWrittentDataSize:
0x1fb3468ff: modelAtURL:key:
0x1fc95018e: purgeCompiledModelDictionary:
0x1fc95048f: updateTmpWeightFilesPath:
0x1fb348240: objectWithIOSurface:statType:
0x1fa11d4bd: printIOSurfaceDataInBytes:
0x1fc94f98d: constructANEModel:modelDirectory:
0x1fb3480e5: modelWithCacheURLIdentifier:
```

## ANEServices
Source: `research/ipsw_ane/raw/ANEServices.dsc.classdump.txt`
```text
@interface ANEServicesLog : NSObject

/* class methods */
+[ANEServicesLog verbose];
+[ANEServicesLog test];
+[ANEServicesLog services];
+[ANEServicesLog handle];

@end

```

## Low/No-ObjC Images
```text
ANECompiler:   - no objc
ANEClientSignals: 0 lines
MilAneflow:   - no objc
```

## Entitlements
Source: `research/ipsw_ane/raw/ane_binary_entitlements.xml`
```text
=== /System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANECompilerService.xpc/Contents/MacOS/ANECompilerService ===
[Dict]
	[Key] com.apple.PerfPowerServices.data-donation
	[Value]
		[Bool] true
	[Key] com.apple.developer.hardened-process
	[Value]
		[Bool] true
	[Key] com.apple.private.coreml.decypt_allowed
	[Value]
		[Bool] true
	[Key] com.apple.private.security.storage.MobileAssetGenerativeModels
	[Value]
		[Bool] true
	[Key] com.apple.rootless.storage.ane_model_cache
	[Value]
		[Bool] true
	[Key] com.apple.rootless.storage.triald
	[Value]
		[Bool] true
	[Key] com.apple.security.exception.mach-lookup.global-name
	[Value]
		[Array]
			[String] com.apple.powerlog.plxpclogger.xpc
			[String] com.apple.PerfPowerTelemetryClientRegistrationService
	[Key] com.apple.security.temporary-exception.mach-lookup.global-name
	[Value]
		[Array]
			[String] com.apple.powerlog.plxpclogger.xpc
			[String] com.apple.PerfPowerTelemetryClientRegistrationService
	[Key] platform-application
	[Value]
		[Bool] true
	[Key] seatbelt-profiles
	[Value]
		[Array]
			[String] ANECompilerService

=== /System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANEStorageMaintainer.xpc/Contents/MacOS/ANEStorageMaintainer ===
[Dict]
	[Key] com.apple.rootless.storage.ane_model_cache
	[Value]
		[Bool] true
	[Key] com.apple.rootless.storage.triald
	[Value]
		[Bool] true
	[Key] platform-application
	[Value]
		[Bool] true

=== /usr/libexec/aned ===
[Dict]
	[Key] com.apple.ANECompilerService.allow
	[Value]
		[Bool] true
	[Key] com.apple.PerfPowerServices.data-donation
	[Value]
		[Bool] true
	[Key] com.apple.ane.iokit-user-access
	[Value]
		[Bool] true
	[Key] com.apple.developer.hardened-process
	[Value]
		[Bool] true
	[Key] com.apple.private.ANEStorageMaintainer.allow
	[Value]
		[Bool] true
	[Key] com.apple.private.kernel.override-cpumon
	[Value]
		[Bool] true
	[Key] com.apple.private.security.storage.MobileAssetGenerativeModels
	[Value]
		[Bool] true
	[Key] com.apple.rootless.storage.ane_model_cache
	[Value]
		[Bool] true
	[Key] com.apple.rootless.storage.triald
	[Value]
		[Bool] true
	[Key] com.apple.security.exception.mach-lookup.global-name
	[Value]
		[Array]
			[String] com.apple.powerlog.plxpclogger.xpc
			[String] com.apple.PerfPowerTelemetryClientRegistrationService
	[Key] com.apple.security.temporary-exception.mach-lookup.global-name
	[Value]
		[Array]
			[String] com.apple.powerlog.plxpclogger.xpc
			[String] com.apple.PerfPowerTelemetryClientRegistrationService
	[Key] platform-application
	[Value]
		[Bool] true
	[Key] seatbelt-profiles
	[Value]
		[Array]
			[String] aned

=== /usr/libexec/aneuserd ===
[Dict]
	[Key] com.apple.ANECompilerService.allow
	[Value]
		[Bool] true
	[Key] com.apple.ane.iokit-user-access
	[Value]
		[Bool] true
	[Key] com.apple.aneuserd.private.allow
	[Value]
		[Bool] true
	[Key] com.apple.private.ANEStorageMaintainer.allow
	[Value]
		[Bool] true
	[Key] com.apple.private.kernel.override-cpumon
	[Value]
		[Bool] true
	[Key] com.apple.rootless.storage.ane_model_cache
	[Value]
		[Bool] true
	[Key] com.apple.rootless.storage.triald
	[Value]
		[Bool] true
	[Key] platform-application
	[Value]
		[Bool] true
	[Key] seatbelt-profiles
	[Value]
		[Array]
			[String] aneuserd

```

## Dynamic Dependencies
Source: `research/ipsw_ane/raw/ane_binary_deps.txt`
```text
=== /System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANECompilerService.xpc/Contents/MacOS/ANECompilerService ===
/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANECompilerService.xpc/Contents/MacOS/ANECompilerService:
	/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/Versions/A/AppleNeuralEngine (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/PrivateFrameworks/PowerLog.framework/Versions/A/PowerLog (compatibility version 1.0.0, current version 1.0.0, weak)
	/usr/lib/libcompression.dylib (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/IOSurface.framework/Versions/A/IOSurface (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/IOKit.framework/Versions/A/IOKit (compatibility version 1.0.0, current version 275.0.0)
	/System/Library/PrivateFrameworks/CoreAnalytics.framework/Versions/A/CoreAnalytics (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/PrivateFrameworks/ANECompiler.framework/Versions/A/ANECompiler (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/PrivateFrameworks/MIL.framework/Versions/A/MIL (compatibility version 3510.2.0, current version 3510.2.1)
	/usr/lib/libMobileGestalt.dylib (compatibility version 1.0.0, current version 1.0.0)
	/usr/lib/libsandbox.1.dylib (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/Security.framework/Versions/A/Security (compatibility version 1.0.0, current version 61901.80.22)
	/System/Library/PrivateFrameworks/Espresso.framework/Versions/A/Espresso (compatibility version 1.0.0, current version 123.0.0)
	/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation (compatibility version 300.0.0, current version 4302.0.0)
	/usr/lib/libobjc.A.dylib (compatibility version 1.0.0, current version 228.0.0)
	/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 2000.67.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)
	/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 4302.0.0)

=== /System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANEStorageMaintainer.xpc/Contents/MacOS/ANEStorageMaintainer ===
/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANEStorageMaintainer.xpc/Contents/MacOS/ANEStorageMaintainer:
	/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/Versions/A/AppleNeuralEngine (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/PrivateFrameworks/PowerLog.framework/Versions/A/PowerLog (compatibility version 1.0.0, current version 1.0.0, weak)
	/usr/lib/libcompression.dylib (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/IOSurface.framework/Versions/A/IOSurface (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/PrivateFrameworks/MIL.framework/Versions/A/MIL (compatibility version 3510.2.0, current version 3510.2.1)
	/System/Library/PrivateFrameworks/ANECompiler.framework/Versions/A/ANECompiler (compatibility version 1.0.0, current version 1.0.0)
	/usr/lib/libMobileGestalt.dylib (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation (compatibility version 300.0.0, current version 4302.0.0)
	/usr/lib/libobjc.A.dylib (compatibility version 1.0.0, current version 228.0.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)
	/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 4302.0.0)

=== /usr/libexec/aned ===
/usr/libexec/aned:
	/System/Library/PrivateFrameworks/ANECompiler.framework/Versions/A/ANECompiler (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/Security.framework/Versions/A/Security (compatibility version 1.0.0, current version 61901.80.22)
	/System/Library/Frameworks/CoreServices.framework/Versions/A/CoreServices (compatibility version 1.0.0, current version 1226.0.0)
	/System/Library/Frameworks/IOSurface.framework/Versions/A/IOSurface (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/IOKit.framework/Versions/A/IOKit (compatibility version 1.0.0, current version 275.0.0)
	/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/Versions/A/AppleNeuralEngine (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation (compatibility version 300.0.0, current version 4302.0.0)
	/usr/lib/libobjc.A.dylib (compatibility version 1.0.0, current version 228.0.0)
	/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 2000.67.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)
	/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 4302.0.0)

=== /usr/libexec/aneuserd ===
/usr/libexec/aneuserd:
	/System/Library/PrivateFrameworks/ANECompiler.framework/Versions/A/ANECompiler (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/Security.framework/Versions/A/Security (compatibility version 1.0.0, current version 61901.80.22)
	/System/Library/Frameworks/CoreServices.framework/Versions/A/CoreServices (compatibility version 1.0.0, current version 1226.0.0)
	/System/Library/Frameworks/IOSurface.framework/Versions/A/IOSurface (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/IOKit.framework/Versions/A/IOKit (compatibility version 1.0.0, current version 275.0.0)
	/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/Versions/A/AppleNeuralEngine (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation (compatibility version 300.0.0, current version 4302.0.0)
	/usr/lib/libobjc.A.dylib (compatibility version 1.0.0, current version 228.0.0)
	/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 2000.67.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)
	/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 4302.0.0)

```

## XPC Metadata
Source: `research/ipsw_ane/raw/ane_xpc_info_plist.txt`
```text
=== /System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANECompilerService.xpc/Contents/Info.plist ===
{
  "CFBundleIdentifier" => "com.apple.ANECompilerService"
  "CFBundleVersion" => "1"
  "DTPlatformBuild" => ""
  "DTXcode" => "2600"
  "BuildMachineOSBuild" => "23A344017"
  "CFBundleInfoDictionaryVersion" => "6.0"
  "DTCompiler" => "com.apple.compilers.llvm.clang.1_0"
  "DTPlatformVersion" => "26.3"
  "DTSDKBuild" => "25D110"
  "CFBundleExecutable" => "ANECompilerService"
  "CFBundleName" => "ANECompilerService"
  "DTPlatformName" => "macosx"
  "DTSDKName" => "macosx26.3.internal"
  "LSMinimumSystemVersion" => "26.3"
  "CFBundleDisplayName" => "ANECompilerService"
  "CFBundlePackageType" => "XPC!"
  "CFBundleSupportedPlatforms" => [
    0 => "MacOSX"
  ]
  "DTXcodeBuild" => "17A6264l"
  "XPCService" => {
    "ServiceType" => "Application"
    "_MultipleInstances" => true
  }
  "CFBundleDevelopmentRegion" => "en"
}

=== /System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANEStorageMaintainer.xpc/Contents/Info.plist ===
{
  "CFBundleName" => "ANEStorageMaintainer"
  "DTCompiler" => "com.apple.compilers.llvm.clang.1_0"
  "DTXcode" => "2600"
  "NSHumanReadableCopyright" => "Copyright © 2021 Apple. All rights reserved."
  "XPCService" => {
    "ServiceType" => "Application"
  }
  "BuildMachineOSBuild" => "23A344017"
  "CFBundlePackageType" => "XPC!"
  "DTPlatformBuild" => ""
  "DTPlatformVersion" => "26.3"
  "DTSDKBuild" => "25D110"
  "CFBundleInfoDictionaryVersion" => "6.0"
  "CFBundleSupportedPlatforms" => [
    0 => "MacOSX"
  ]
  "CFBundleVersion" => "1"
  "LSMinimumSystemVersion" => "26.3"
  "CFBundleDisplayName" => "ANEStorageMaintainer"
  "CFBundleIdentifier" => "com.apple.private.ANEStorageMaintainer"
  "DTPlatformName" => "macosx"
  "DTSDKName" => "macosx26.3.internal"
  "DTXcodeBuild" => "17A6264l"
  "CFBundleDevelopmentRegion" => "en"
  "CFBundleExecutable" => "ANEStorageMaintainer"
}

```

## Raw Artifacts
```text
research/ipsw_ane/raw/ANEClientSignals.dsc.classdump.txt
research/ipsw_ane/raw/ANEClientSignals.dsc.err
research/ipsw_ane/raw/ANEClientSignals.dyld.macho.objc.err
research/ipsw_ane/raw/ANEClientSignals.dyld.macho.objc.txt
research/ipsw_ane/raw/ANEClientSignals.dyld.macho.objc_refs.err
research/ipsw_ane/raw/ANEClientSignals.dyld.macho.objc_refs.txt
research/ipsw_ane/raw/ANECompiler.dsc.classdump.txt
research/ipsw_ane/raw/ANECompiler.dsc.err
research/ipsw_ane/raw/ANECompiler.dyld.macho.objc.err
research/ipsw_ane/raw/ANECompiler.dyld.macho.objc.txt
research/ipsw_ane/raw/ANECompiler.dyld.macho.objc_refs.err
research/ipsw_ane/raw/ANECompiler.dyld.macho.objc_refs.txt
research/ipsw_ane/raw/ANECompilerService.classdump.protocol_methods.txt
research/ipsw_ane/raw/ANECompilerService.classdump.re.txt
research/ipsw_ane/raw/ANECompilerService.classdump.refs.txt
research/ipsw_ane/raw/ANECompilerService.classdump.txt
research/ipsw_ane/raw/ANECompilerService.macho.objc.err
research/ipsw_ane/raw/ANECompilerService.macho.objc.txt
research/ipsw_ane/raw/ANECompilerService.macho.objc_refs.err
research/ipsw_ane/raw/ANECompilerService.macho.objc_refs.txt
research/ipsw_ane/raw/ANEServices.dsc.classdump.txt
research/ipsw_ane/raw/ANEServices.dsc.err
research/ipsw_ane/raw/ANEServices.dyld.macho.objc.err
research/ipsw_ane/raw/ANEServices.dyld.macho.objc.txt
research/ipsw_ane/raw/ANEServices.dyld.macho.objc_refs.err
research/ipsw_ane/raw/ANEServices.dyld.macho.objc_refs.txt
research/ipsw_ane/raw/ANEServices.dyld.objc.class.txt
research/ipsw_ane/raw/ANEServices.dyld.objc.proto.txt
research/ipsw_ane/raw/ANEServices.dyld.objc.sel.txt
research/ipsw_ane/raw/ANEStorageMaintainer.classdump.protocol_methods.txt
research/ipsw_ane/raw/ANEStorageMaintainer.classdump.re.txt
research/ipsw_ane/raw/ANEStorageMaintainer.classdump.refs.txt
research/ipsw_ane/raw/ANEStorageMaintainer.classdump.txt
research/ipsw_ane/raw/ANEStorageMaintainer.macho.objc.err
research/ipsw_ane/raw/ANEStorageMaintainer.macho.objc.txt
research/ipsw_ane/raw/ANEStorageMaintainer.macho.objc_refs.err
research/ipsw_ane/raw/ANEStorageMaintainer.macho.objc_refs.txt
research/ipsw_ane/raw/AppleNeuralEngine.dsc.classdump.protocol_methods.txt
research/ipsw_ane/raw/AppleNeuralEngine.dsc.classdump.re.txt
research/ipsw_ane/raw/AppleNeuralEngine.dsc.classdump.refs.txt
research/ipsw_ane/raw/AppleNeuralEngine.dsc.classdump.txt
research/ipsw_ane/raw/AppleNeuralEngine.dsc.err
research/ipsw_ane/raw/AppleNeuralEngine.dyld.macho.objc.err
research/ipsw_ane/raw/AppleNeuralEngine.dyld.macho.objc.txt
research/ipsw_ane/raw/AppleNeuralEngine.dyld.macho.objc_refs.err
research/ipsw_ane/raw/AppleNeuralEngine.dyld.macho.objc_refs.txt
research/ipsw_ane/raw/AppleNeuralEngine.dyld.objc.class.txt
research/ipsw_ane/raw/AppleNeuralEngine.dyld.objc.proto.txt
research/ipsw_ane/raw/AppleNeuralEngine.dyld.objc.sel.key.txt
research/ipsw_ane/raw/AppleNeuralEngine.dyld.objc.sel.txt
research/ipsw_ane/raw/MilAneflow.dsc.classdump.txt
research/ipsw_ane/raw/MilAneflow.dsc.err
research/ipsw_ane/raw/MilAneflow.dyld.macho.objc.err
research/ipsw_ane/raw/MilAneflow.dyld.macho.objc.txt
research/ipsw_ane/raw/MilAneflow.dyld.macho.objc_refs.err
research/ipsw_ane/raw/MilAneflow.dyld.macho.objc_refs.txt
research/ipsw_ane/raw/ParavirtualizedANE.dsc.classdump.re.txt
research/ipsw_ane/raw/ParavirtualizedANE.dsc.classdump.refs.txt
research/ipsw_ane/raw/ParavirtualizedANE.dsc.classdump.txt
research/ipsw_ane/raw/ParavirtualizedANE.dsc.err
research/ipsw_ane/raw/ParavirtualizedANE.dyld.macho.objc.err
research/ipsw_ane/raw/ParavirtualizedANE.dyld.macho.objc.txt
research/ipsw_ane/raw/ParavirtualizedANE.dyld.macho.objc_refs.err
research/ipsw_ane/raw/ParavirtualizedANE.dyld.macho.objc_refs.txt
research/ipsw_ane/raw/ParavirtualizedANE.dyld.objc.class.txt
research/ipsw_ane/raw/ParavirtualizedANE.dyld.objc.proto.txt
research/ipsw_ane/raw/ParavirtualizedANE.dyld.objc.sel.key.txt
research/ipsw_ane/raw/ParavirtualizedANE.dyld.objc.sel.txt
research/ipsw_ane/raw/ane_binary_deps.txt
research/ipsw_ane/raw/ane_binary_entitlements.xml
research/ipsw_ane/raw/ane_xpc_info_plist.txt
research/ipsw_ane/raw/aned.arm64e.classdump.err
research/ipsw_ane/raw/aned.arm64e.classdump.txt
research/ipsw_ane/raw/aned.arm64e.macho.objc.err
research/ipsw_ane/raw/aned.arm64e.macho.objc.txt
research/ipsw_ane/raw/aned.arm64e.macho.objc_refs.err
research/ipsw_ane/raw/aned.arm64e.macho.objc_refs.txt
research/ipsw_ane/raw/aned.classdump.re.txt
research/ipsw_ane/raw/aned.classdump.refs.txt
research/ipsw_ane/raw/aned.classdump.txt
research/ipsw_ane/raw/aned.macho.objc.err
research/ipsw_ane/raw/aned.macho.objc.txt
research/ipsw_ane/raw/aned.macho.objc_refs.err
research/ipsw_ane/raw/aned.macho.objc_refs.txt
research/ipsw_ane/raw/aned.x86_64.classdump.err
research/ipsw_ane/raw/aned.x86_64.classdump.txt
research/ipsw_ane/raw/aned.x86_64.macho.objc.err
research/ipsw_ane/raw/aned.x86_64.macho.objc.txt
research/ipsw_ane/raw/aned.x86_64.macho.objc_refs.err
research/ipsw_ane/raw/aned.x86_64.macho.objc_refs.txt
research/ipsw_ane/raw/aneuserd.arm64e.classdump.err
research/ipsw_ane/raw/aneuserd.arm64e.classdump.txt
research/ipsw_ane/raw/aneuserd.arm64e.macho.objc.err
research/ipsw_ane/raw/aneuserd.arm64e.macho.objc.txt
research/ipsw_ane/raw/aneuserd.arm64e.macho.objc_refs.err
research/ipsw_ane/raw/aneuserd.arm64e.macho.objc_refs.txt
research/ipsw_ane/raw/aneuserd.classdump.re.txt
research/ipsw_ane/raw/aneuserd.classdump.refs.txt
research/ipsw_ane/raw/aneuserd.classdump.txt
research/ipsw_ane/raw/aneuserd.macho.objc.err
research/ipsw_ane/raw/aneuserd.macho.objc.txt
research/ipsw_ane/raw/aneuserd.macho.objc_refs.err
research/ipsw_ane/raw/aneuserd.macho.objc_refs.txt
research/ipsw_ane/raw/aneuserd.x86_64.classdump.err
research/ipsw_ane/raw/aneuserd.x86_64.classdump.txt
research/ipsw_ane/raw/aneuserd.x86_64.macho.objc.err
research/ipsw_ane/raw/aneuserd.x86_64.macho.objc.txt
research/ipsw_ane/raw/aneuserd.x86_64.macho.objc_refs.err
research/ipsw_ane/raw/aneuserd.x86_64.macho.objc_refs.txt
research/ipsw_ane/raw/dyld_objc_all.txt
research/ipsw_ane/raw/dyld_objc_ane_only.txt
research/ipsw_ane/raw/environment.txt
research/ipsw_ane/raw/summary_stats.txt
```
