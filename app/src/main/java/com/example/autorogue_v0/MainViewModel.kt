package com.example.autorogue_v0

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel

// --- ViewModel to hold inference state ---
class MainViewModel : ViewModel() {
    var inferenceResult by mutableStateOf("Waiting for input...")
    var confidenceLevel by mutableStateOf(0.0f)
    var beepEnabled by mutableStateOf(false)
}