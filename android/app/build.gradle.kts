plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// -PbundleModels=true packs the model files (from ../../build/apk-assets) into the APK so it is
// self-contained; without it the app expects them pushed to its external files dir via adb.
val bundleModels = (project.findProperty("bundleModels") as String?)?.toBoolean() ?: false
// -Pabis=arm64-v8a,x86_64 adds the emulator ABI (default: phones only).
val abiList = ((project.findProperty("abis") as String?) ?: "arm64-v8a").split(",")

android {
    namespace = "dev.local.miravoice"
    compileSdk = 36
    ndkVersion = "29.0.14206865"

    defaultConfig {
        applicationId = "dev.local.miravoice"
        minSdk = 29
        targetSdk = 36
        versionCode = 5
        versionName = "0.4.1"

        ndk { abiFilters += abiList }
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_SHARED_LIBS=ON",
                    "-DLLAMA_BUILD_COMMON=OFF",
                    "-DLLAMA_BUILD_TESTS=OFF",
                    "-DLLAMA_BUILD_TOOLS=OFF",
                    "-DLLAMA_BUILD_EXAMPLES=OFF",
                    "-DLLAMA_BUILD_SERVER=OFF",
                    "-DLLAMA_BUILD_APP=OFF",
                    "-DLLAMA_OPENSSL=OFF",
                    "-DLLAMA_CURL=OFF",
                    "-DGGML_NATIVE=OFF",
                    // one CPU backend per ARM feature level, picked at runtime for the phone's cores
                    "-DGGML_BACKEND_DL=ON",
                    "-DGGML_CPU_ALL_VARIANTS=ON",
                    "-DGGML_LLAMAFILE=OFF",
                )
            }
        }
    }

    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }

    buildTypes {
        // The debug build is also optimised: native code is always Release (see CMake args).
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlin { jvmToolchain(17) }

    buildFeatures {
        viewBinding = true
        buildConfig = true
    }

    androidResources {
        // models are mmap'd / streamed out of the APK; never compress them
        noCompress += listOf("gguf", "onnx", "bin")
    }
    if (bundleModels) {
        sourceSets["main"].assets.srcDirs("../../build/apk-assets")
    }
    packaging {
        jniLibs { useLegacyPackaging = true }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.17.0")
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("com.google.android.material:material:1.13.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.0")
}
