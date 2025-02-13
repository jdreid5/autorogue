package com.example.autorogue_v0

import android.content.Context
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack

// --- BeepPlayer using AudioTrack
class BeepPlayer(private val context: Context) {
    private var audioTrack: AudioTrack? = null

    // Generate and plays a 1000Hz sine tone
    fun playBeep() {
        // Stop current beep if it exists
        stopBeep()
        val sampleRate = 44100
        val durationMs = 300
        val numSamples = (durationMs * sampleRate) / 1000
        val fadeOutSamples = (0.2 * numSamples).toInt()
        val samples = ShortArray(numSamples)
        val freqOfTone = 1000.0
        for (i in samples.indices) {
            // Compute the fade out
            val amplitude = if (i >= numSamples - fadeOutSamples) {
                (numSamples - i).toDouble() / fadeOutSamples
            } else {
                1.0
            }
            // Generate a sine wave sample
            samples[i] = (Short.MAX_VALUE * amplitude * Math.sin(2 * Math.PI * i / (sampleRate / freqOfTone))).toInt().toShort()
        }
        // Create static audioTrack instance
        audioTrack = AudioTrack(
            AudioManager.STREAM_MUSIC,
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            numSamples * 2,
            AudioTrack.MODE_STATIC
        )
        audioTrack?.write(samples, 0, numSamples)
        audioTrack?.play()
    }

    fun stopBeep() {
        audioTrack?.stop()
        audioTrack?.release()
        audioTrack = null
    }
}