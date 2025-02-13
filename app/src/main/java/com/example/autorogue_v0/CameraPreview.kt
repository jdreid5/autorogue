package com.example.autorogue_v0

import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import java.util.concurrent.Executors

// --- Composable for Camera Preview and Image Analysis ---
@Composable
fun CameraPreview(
    viewModel: MainViewModel,
    yuvToRgbConverter: YuvToRgbConverter,
    tfliteInterpreter: Interpreter,
    imageProcessor: ImageProcessor,
    frameInterval: Long,
    inferenceDispatcher: CoroutineDispatcher,
    beepPlayer: BeepPlayer
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val previewView = remember { PreviewView(context) }

    LaunchedEffect(cameraProviderFuture) {
        val cameraProvider = cameraProviderFuture.get()

        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        // Reduce resolution to ease processing.
        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(640, 480))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        imageAnalysis.setAnalyzer(Executors.newSingleThreadExecutor(), object : ImageAnalysis.Analyzer {
            // Reuse a Bitmap buffer for YUV-to-RGB conversion.
            private var rgbBuffer: Bitmap? = null
            private var lastFrameTime = 0L

            override fun analyze(imageProxy: ImageProxy) {
                val currentTime = System.currentTimeMillis()
                if (currentTime - lastFrameTime < frameInterval) {
                    imageProxy.close()
                    return
                }
                lastFrameTime = currentTime

                // Ensure the rgbBuffer matches the frame dimensions.
                if (rgbBuffer == null ||
                    rgbBuffer?.width != imageProxy.width ||
                    rgbBuffer?.height != imageProxy.height
                ) {
                    rgbBuffer = Bitmap.createBitmap(
                        imageProxy.width,
                        imageProxy.height,
                        Bitmap.Config.ARGB_8888
                    )
                }

                // Run inference on the dedicated inferenceDispatcher.
                CoroutineScope(inferenceDispatcher).launch {
                    try {
                        // Convert YUV image to an RGB Bitmap.
                        yuvToRgbConverter.yuvToRgb(imageProxy, rgbBuffer!!)
                        // Preprocess the image.
                        val tensorImage = TensorImage.fromBitmap(rgbBuffer)
                        val processedImage = imageProcessor.process(tensorImage)
                        val output = Array(1) { FloatArray(1) }
                        // Run inference.
                        tfliteInterpreter.run(processedImage.buffer, output)
                        val confidence = output[0][0]
                        val resultText = if (confidence > 0.5f) "Leaf roll detected" else "No leaf roll detected"
                        if (confidence > 0.75f && viewModel.beepEnabled) {
                            beepPlayer.playBeep()
                        }
                        // Update UI on the main thread.
                        withContext(Dispatchers.Main) {
                            viewModel.inferenceResult = resultText
                            viewModel.confidenceLevel = confidence
                        }
                    } catch (e: Exception) {
                        Log.e("ImageAnalysis", "Error during analysis", e)
                    } finally {
                        imageProxy.close()
                    }
                }
            }
        })

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalysis
            )
        } catch (exc: Exception) {
            Log.e("CameraXApp", "Use case binding failed", exc)
        }
    }

    AndroidView(
        factory = { previewView },
        modifier = Modifier.fillMaxSize()
    )
}