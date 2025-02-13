package com.example.autorogue_v0

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.example.autorogue_v0.ui.theme.Autorogue_v0Theme
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import java.util.Locale

// --- Main Activity ---
class MainActivity : ComponentActivity() {

    // Interpreter and BeepPlayer are initialized on the dedicated inference thread.
    private lateinit var tfliteInterpreter: Interpreter
    private lateinit var beepPlayer: BeepPlayer

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

        // Initialize beepPlayer
        beepPlayer = BeepPlayer(this)

        // Create and cache the ImageProcessor for preprocessing.
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(224, 298))
            .add(ResizeOp(224, 298, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()

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
                                frameInterval = frameInterval,
                                inferenceDispatcher = inferenceDispatcher,
                                beepPlayer = beepPlayer
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
                                text = String.format(Locale.UK, "Confidence: %.2f%%", viewModel.confidenceLevel * 100),
                                fontSize = 18.sp,
                                color = Color.White,
                                modifier = Modifier.padding(16.dp)
                            )
                        }
                        Column(
                            modifier = Modifier
                                .align(Alignment.BottomCenter)
                                .background(Color(0xAA000000))
                                .padding(16.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Button(
                                onClick = { viewModel.beepEnabled = !viewModel.beepEnabled },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = Color.Transparent
                                )
                            ) {
                                Text(
                                    text = if (viewModel.beepEnabled) "Disable Sound" else "Enable Sound",
                                    color = Color.White,
                                    fontSize = 18.sp
                                )
                            }
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
        inferenceDispatcher.cancel() // Cancel the dispatcher to clean up resources.
        inferenceExecutor.shutdown()
    }

    override fun onStop() {
        super.onStop()
        // Stop any currently playing beep sound.
        beepPlayer.stopBeep()
    }
}