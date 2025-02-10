package com.example.autorogue_v0

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.AudioAttributes
import android.media.SoundPool
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicYuvToRGB
import android.renderscript.Type
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModel
import com.example.autorogue_v0.ui.theme.Autorogue_v0Theme
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executors

// --- ViewModel to hold inference state ---
class MainViewModel : ViewModel() {
    var inferenceResult by mutableStateOf("Waiting for input...")
    var confidenceLevel by mutableStateOf(0.0f)
}

// --- Main Activity ---
class MainActivity : ComponentActivity() {

    // Interpreter and SoundPool are initialized on the dedicated inference thread.
    private lateinit var tfliteInterpreter: Interpreter
    private lateinit var soundPool: SoundPool
    private var beepStreamId: Int = 0
    private var beepSoundId: Int = 0

    // Preprocessing components.
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var yuvToRgbConverter: YuvToRgbConverter

    // Frame rate control: process at most one frame every 300ms.
    private var lastProcessedTime = 0L
    private val frameInterval = 300L

    // Dedicated single-threaded executor and its coroutine dispatcher.
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val inferenceDispatcher = inferenceExecutor.asCoroutineDispatcher()

    // Simple ViewModel to hold inference state.
    private val viewModel: MainViewModel by viewModels()

    // Flag for camera permission.
    private var cameraPermissionGranted by mutableStateOf(false)

    // Flag to signal that the interpreter is initialized.
    private var interpreterReady by mutableStateOf(false)

    // Helper function to update the beepStreamId.
    fun updateBeepStreamId(streamId: Int) {
        beepStreamId = streamId
    }

    // Helper: Check GPU delegate availability.
    object GpuDelegateHelper {
        fun isGpuDelegateAvailable(): Boolean {
            return try {
                val delegate = org.tensorflow.lite.gpu.GpuDelegate()
                delegate.close()
                true
            } catch (e: NoClassDefFoundError) {
                false
            } catch (e: Exception) {
                false
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize our YUV-to-RGB converter.
        yuvToRgbConverter = YuvToRgbConverter(this)

        // Initialize the TFLite interpreter on the dedicated inference thread.
        initializeInterpreterOnInferenceThread()

        // Create and cache the ImageProcessor for preprocessing.
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(224, 298))
            .add(ResizeOp(224, 298, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()

        // Initialize SoundPool to play a beep when confidence is high.
        initializeSoundPool()

        enableEdgeToEdge()
        setContent {
            Autorogue_v0Theme {
                Scaffold(modifier = Modifier.fillMaxSize()) { padding ->
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(padding)
                    ) {
                        // Only show the CameraPreview if the camera permission is granted and the interpreter is ready.
                        if (cameraPermissionGranted && interpreterReady) {
                            CameraPreview(
                                viewModel = viewModel,
                                yuvToRgbConverter = yuvToRgbConverter,
                                tfliteInterpreter = tfliteInterpreter,
                                imageProcessor = imageProcessor,
                                soundPool = soundPool,
                                beepSoundId = beepSoundId,
                                frameInterval = frameInterval,
                                inferenceDispatcher = inferenceDispatcher,
                                onBeepPlayed = { streamId -> updateBeepStreamId(streamId) }
                            )
                        } else {
                            // Show a loading indicator if we are still initializing.
                            Column(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .background(Color(0xFF000000)),
                                verticalArrangement = Arrangement.Center,
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                CircularProgressIndicator(color = Color.White)
                                Spacer(modifier = Modifier.height(16.dp))
                                Text(
                                    text = "Initializing...",
                                    fontSize = 20.sp,
                                    color = Color.White
                                )
                            }
                        }
                        // Overlay UI for inference results.
                        Column(
                            modifier = Modifier
                                .align(Alignment.TopCenter)
                                .background(Color(0xAA000000))
                                .padding(16.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Text(
                                text = viewModel.inferenceResult,
                                fontSize = 20.sp,
                                color = Color.White,
                                modifier = Modifier.padding(16.dp)
                            )
                            Text(
                                text = String.format("Confidence: %.2f%%", viewModel.confidenceLevel * 100),
                                fontSize = 18.sp,
                                color = Color.White,
                                modifier = Modifier.padding(16.dp)
                            )
                        }
                    }
                }
            }
        }

        // Check and request camera permission.
        checkCameraPermission()
    }

    private fun initializeInterpreterOnInferenceThread() {
        // Launch a coroutine on the dedicated inferenceDispatcher to initialize the interpreter.
        CoroutineScope(inferenceDispatcher).launch {
            val model = loadModelFile("autorogue_v0.tflite")
            val options = Interpreter.Options()
            try {
                // Attempt to create GPU delegate if available.
                if (GpuDelegateHelper.isGpuDelegateAvailable()) {
                    val gpuDelegate = org.tensorflow.lite.gpu.GpuDelegate()
                    options.addDelegate(gpuDelegate)
                } else {
                    options.setUseNNAPI(true)
                }
            } catch (e: Throwable) {
                options.setUseNNAPI(true)
            }
            tfliteInterpreter = Interpreter(model, options)
            // Mark the interpreter as ready on the main thread.
            withContext(Dispatchers.Main) {
                interpreterReady = true
            }
        }
    }

    private fun initializeSoundPool() {
        val audioAttributes = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
            .build()

        soundPool = SoundPool.Builder()
            .setMaxStreams(1)
            .setAudioAttributes(audioAttributes)
            .build()

        // Load the beep sound from res/raw/beep.mp3.
        beepSoundId = soundPool.load(this, R.raw.beep, 1)
    }

    private fun loadModelFile(modelFilename: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelFilename)
        val fileInputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED -> {
                cameraPermissionGranted = true
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            cameraPermissionGranted = isGranted
            if (!isGranted) {
                Toast.makeText(
                    this,
                    "Camera permission is required to use this feature",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }

    override fun onDestroy() {
        super.onDestroy()
        soundPool.release()
        inferenceDispatcher.cancel() // Cancel the dispatcher to clean up resources.
        inferenceExecutor.shutdown()
    }

    override fun onStop() {
        super.onStop()
        // Stop any currently playing beep sound.
        // Stop the beep if one is playing.
        if (beepStreamId != 0) {
            soundPool.stop(beepStreamId)
            beepStreamId = 0
        }
        soundPool.autoPause()
    }

    override fun onStart() {
        super.onStart()
        soundPool.autoResume()
    }
}

// --- Composable for Camera Preview and Image Analysis ---
@Composable
fun CameraPreview(
    viewModel: MainViewModel,
    yuvToRgbConverter: YuvToRgbConverter,
    tfliteInterpreter: Interpreter,
    imageProcessor: ImageProcessor,
    soundPool: SoundPool,
    beepSoundId: Int,
    frameInterval: Long,
    inferenceDispatcher: CoroutineDispatcher,
    onBeepPlayed: (Int) -> Unit
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
                        if (confidence > 0.75f) {
                            val streamId = soundPool.play(beepSoundId, 1f, 1f, 0, 0, 1f)
                            onBeepPlayed(streamId)
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

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(text = "Hello $name!", modifier = modifier)
}

@Composable
fun GreetingPreview() {
    Autorogue_v0Theme {
        Greeting("Android")
    }
}

// --- YUV to RGB Converter using RenderScript ---
// (Note: RenderScript is deprecated in newer Android versions; consider alternatives if needed)
class YuvToRgbConverter(private val context: Context) {
    private val rs: RenderScript = RenderScript.create(context)
    private val yuvToRgbIntrinsic: ScriptIntrinsicYuvToRGB =
        ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))

    fun yuvToRgb(image: ImageProxy, output: Bitmap) {
        val nv21 = yuv420888ToNv21(image)
        val yuvType = Type.Builder(rs, Element.U8(rs)).setX(nv21.size)
        val inAllocation = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT)
        inAllocation.copyFrom(nv21)
        yuvToRgbIntrinsic.setInput(inAllocation)
        val outAllocation = Allocation.createFromBitmap(rs, output)
        yuvToRgbIntrinsic.forEach(outAllocation)
        outAllocation.copyTo(output)
    }

    // Convert ImageProxy in YUV_420_888 format to an NV21 byte array.
    private fun yuv420888ToNv21(image: ImageProxy): ByteArray {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val uvSize = width * height / 4
        val nv21 = ByteArray(ySize + uvSize * 2)

        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val yRowStride = image.planes[0].rowStride
        val uvRowStride = image.planes[1].rowStride

        var pos = 0
        // Copy Y plane.
        for (row in 0 until height) {
            yBuffer.position(row * yRowStride)
            yBuffer.get(nv21, pos, width)
            pos += width
        }
        // Interleave V and U (NV21 expects V then U).
        for (row in 0 until height / 2) {
            uBuffer.position(row * uvRowStride)
            vBuffer.position(row * uvRowStride)
            for (col in 0 until width / 2) {
                nv21[pos++] = vBuffer.get()
                nv21[pos++] = uBuffer.get()
            }
        }
        return nv21
    }
}