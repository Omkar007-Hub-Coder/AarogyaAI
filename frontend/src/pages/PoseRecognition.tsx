import { useState, useRef } from 'react'
import { recognizePoseFromImage } from '../api/client'
import type { PoseRecognitionResult } from '../types'

export default function PoseRecognition() {
  const [result, setResult] = useState<PoseRecognitionResult | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  function handleFile(file: File) {
    const reader = new FileReader()
    reader.onload = e => setPreview(e.target?.result as string)
    reader.readAsDataURL(file)
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    handleFile(file)
    setResult(null)
    setError(null)
    setLoading(true)
    try {
      const res = await recognizePoseFromImage(file)
      setResult(res)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      if (msg.includes('503') || msg.includes('404')) {
        setError('Pose recognition model is not yet available. The EfficientNetB0 classifier requires training to complete first.')
      } else {
        setError(`Recognition failed: ${msg}`)
      }
    } finally {
      setLoading(false)
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      handleFile(file)
      // Trigger recognition via file input simulation
      const dt = new DataTransfer()
      dt.items.add(file)
      if (fileRef.current) {
        fileRef.current.files = dt.files
        fileRef.current.dispatchEvent(new Event('change', { bubbles: true }))
      }
    }
  }

  return (
    <div className="max-w-lg mx-auto pb-12">
      <h2 className="text-2xl font-bold text-gray-900 mb-1">Pose Recognition</h2>
      <p className="text-sm text-gray-500 mb-6">
        Upload a yoga pose image to classify it using the EfficientNetB0 model trained on Yoga-82 (82 classes, real dataset).
      </p>

      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs text-amber-700 mb-4">
        <strong>Note:</strong> This uses a real ML model (EfficientNetB0, Yoga-82 dataset). Pose recognition is for educational/research purposes only — not a medical assessment tool.
      </div>

      {/* Upload area */}
      <div
        className="bg-white border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-green-400 transition-colors cursor-pointer"
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
        onClick={() => fileRef.current?.click()}
      >
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleUpload}
        />
        {preview ? (
          <img src={preview} alt="Preview" className="max-h-48 mx-auto rounded-lg object-contain mb-3" />
        ) : (
          <div className="text-gray-400 space-y-2">
            <div className="text-4xl">🖼️</div>
            <p className="text-sm">Drop an image here or click to browse</p>
            <p className="text-xs">JPG, PNG, WebP supported</p>
          </div>
        )}
        {preview && <p className="text-xs text-gray-400 mt-1">Click to change image</p>}
      </div>

      {loading && (
        <div className="mt-4 flex items-center justify-center gap-2 text-gray-500 text-sm">
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
          </svg>
          Classifying pose…
        </div>
      )}

      {error && (
        <div className="mt-4 bg-red-50 border border-red-200 text-red-700 rounded-lg p-4 text-sm">{error}</div>
      )}

      {result && (
        <div className="mt-4 bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
          <h3 className="font-semibold text-gray-800 mb-3">Classification Result</h3>
          <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-4">
            <p className="text-xs text-gray-500 mb-0.5">Predicted pose</p>
            <p className="font-bold text-green-800 text-lg">{result.predicted_pose.replace(/_/g, ' ')}</p>
            <div className="flex items-center gap-2 mt-1">
              <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                <div className="bg-green-500 h-1.5 rounded-full" style={{ width: `${(result.confidence*100).toFixed(0)}%` }} />
              </div>
              <span className="text-xs font-medium text-green-700">{(result.confidence*100).toFixed(1)}%</span>
            </div>
          </div>
          <div>
            <p className="text-xs font-medium text-gray-600 mb-2">Top-5 predictions</p>
            <div className="space-y-1.5">
              {result.top_k.map((item, i) => (
                <div key={item.pose} className="flex items-center gap-2 text-xs">
                  <span className="text-gray-400 w-4 text-right">{i+1}.</span>
                  <span className="flex-1 text-gray-700 truncate">{item.pose.replace(/_/g, ' ')}</span>
                  <div className="w-24 bg-gray-100 rounded-full h-1.5">
                    <div className="bg-green-400 h-1.5 rounded-full" style={{ width: `${(item.confidence*100).toFixed(0)}%` }} />
                  </div>
                  <span className="text-gray-500 w-10 text-right">{(item.confidence*100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
          <p className="text-xs text-gray-400 mt-3">
            Model: EfficientNetB0 trained on Yoga-82 (82 classes, 11,608 real training images). Top-1 test accuracy: ongoing training.
          </p>
        </div>
      )}
    </div>
  )
}
