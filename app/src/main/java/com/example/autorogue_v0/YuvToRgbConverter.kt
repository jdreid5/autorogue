package com.example.autorogue_v0

import android.content.Context
import android.graphics.Bitmap
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicYuvToRGB
import android.renderscript.Type
import androidx.camera.core.ImageProxy

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